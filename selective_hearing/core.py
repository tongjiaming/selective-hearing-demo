"""Model-independent identity and playback rules. No audio device access."""
from dataclasses import dataclass, field
from itertools import product
from threading import Lock

import numpy as np
from scipy.signal import resample_poly
from math import gcd
import time

from .feature_bank import FeatureBank, MIN_SPEECH_SECONDS


def mono(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("音频必须为有限数值的单声道或双声道波形")
    return audio


def resample(audio, source_rate, target_rate):
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    divisor = gcd(int(source_rate), int(target_rate))
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("无效说话人特征")
    return vector / norm


@dataclass
class Speaker:
    id: str
    vector: np.ndarray
    observations: int = 1
    seconds: float = 0.0
    learned_vector: np.ndarray | None = None
    learned_seconds: float = 0.0
    learned_observations: int = 0

    def similarity(self, vector):
        anchor = float(vector @ self.vector)
        # A drifting learned centroid cannot overrule a weak anchor match on
        # its own. Wider voice variation needs multiple reliable neighbors.
        return max(anchor, float(vector @ self.learned_vector)
                   if self.learned_vector is not None and anchor >= .35 else -1.0)

    def observe(self, vector, seconds, learn):
        self.observations += 1
        self.seconds += seconds
        # Keep the original reference intact. Learning cannot walk indefinitely
        # toward an unrelated voice through a chain of borderline matches.
        if not learn or float(vector @ self.vector) < 0.55 or self.similarity(vector) < 0.55:
            return
        center = self.learned_vector if self.learned_vector is not None else self.vector
        weight = min(self.learned_seconds, 20.0) if self.learned_vector is not None else 2.0
        added = min(seconds, 3.0)
        updated = unit(center * weight + vector * added)
        if float(updated @ self.vector) < 0.65:
            return
        self.learned_vector = updated
        self.learned_seconds = weight + added
        self.learned_observations += 1

    @property
    def ready(self):
        return self.observations >= 2 and self.seconds >= 1.5


@dataclass(eq=False)
class VoiceCandidate:
    vector: np.ndarray
    seconds: float
    first_block: int
    last_block: int
    observations: int = 1
    borderline: bool = False
    consistent_observations: int = 1
    recent_vectors: list = field(default_factory=list, repr=False)
    seed_vector: np.ndarray | None = field(default=None, repr=False)
    seed_seconds: float = 0.0
    seed_context_seconds: float | None = None

    def __post_init__(self):
        if self.seed_vector is None:
            self.seed_vector = self.vector.copy()
            self.seed_seconds = self.seconds
        if self.seed_context_seconds is None:
            self.seed_context_seconds = self.seed_seconds

    def observe(self, vector, seconds, block, borderline, embedding_seconds=None):
        quality = seconds if embedding_seconds is None else embedding_seconds
        if (seconds >= .75 and (self.seed_seconds < .75 or quality >= self.seed_context_seconds)
                or seconds < .75 and self.seed_seconds < .75 and seconds >= self.seed_seconds):
            self.seed_vector = vector.copy()
            self.seed_seconds = seconds
            self.seed_context_seconds = quality
        self.vector = unit(self.vector * self.seconds + vector * seconds)
        self.seconds += seconds
        self.observations += 1
        self.last_block = block
        self.borderline |= borderline
        self.recent_vectors = (self.recent_vectors + [vector.copy()])[-3:]
        self.consistent_observations = 1
        for count in range(2, len(self.recent_vectors) + 1):
            tail = self.recent_vectors[-count:]
            if all(float(a @ b) >= .75 for i, a in enumerate(tail) for b in tail[i + 1:]):
                self.consistent_observations = count
            else:
                break


class SpeakerRegistry:
    """Optionally persistent identities; each separator block has at most two sources.

    Joint assignment handles separator permutations and duplicate outputs.
    First voices are selectable immediately; ambiguous novelty accumulates evidence.
    Strong later evidence can reassign samples and retire duplicate drafts.
    eager=False retains the historical deferred-discovery comparison mode.
    """
    def __init__(self, threshold=0.55, new_threshold=0.30, margin=0.10, min_new_seconds=1.0,
                 short_threshold=None, candidate_threshold=0.50,
                 candidate_blocks=6, max_candidates=12, store=None, learning=True, eager=True):
        self.threshold = threshold
        short_threshold = (0.40 if eager else 0.45) if short_threshold is None else short_threshold
        self.short_threshold = short_threshold
        self.new_threshold = new_threshold
        self.margin = margin
        self.min_new_seconds = min_new_seconds
        self.candidate_threshold = candidate_threshold
        self.candidate_blocks = candidate_blocks
        self.max_candidates = max_candidates
        self.store = store
        self.learning = learning
        self.eager = eager
        self.speakers = store.load_speakers() if store is not None else []
        self.aliases = store.load_aliases() if store is not None else {}
        self.conflicts = store.load_conflicts() if store is not None else set()
        self.next_identity = max((int(s.id.split('_')[-1]) for s in self.speakers), default=0) + 1
        self.last_merges = []
        self.candidates = []
        self.block = 0
        self.decisions = []
        self.discovery_ceiling = min(short_threshold, threshold)
        self.feature_bank = FeatureBank(store.load_samples() if store is not None else ())
        self.last_corrections = []
        self._restore_candidates()

    def _restore_candidates(self):
        # Recent unresolved evidence survives an app restart. Reconstruct only
        # candidates here: loading a profile must not create people by itself.
        newest = max((s.batch_id for s in self.feature_bank.samples), default=0)
        for sample in self.feature_bank.samples:
            if sample.speaker_id is not None or sample.batch_id < newest - self.max_candidate_age:
                continue
            nearest = max(self._scores(sample.vector, self.speakers), default=-1.0)
            if nearest >= self.discovery_ceiling:
                continue
            block = sample.batch_id - newest
            self.candidates = [c for c in self.candidates
                               if block - c.last_block <= self.candidate_blocks
                               and block - c.first_block <= self.max_candidate_age]
            matches = sorted(((float(sample.vector @ c.vector), c) for c in self.candidates),
                             key=lambda pair: pair[0], reverse=True)
            if matches and matches[0][0] >= self.candidate_threshold:
                if matches[0][1].last_block == block:
                    continue
                if len(matches) > 1 and matches[0][0] - matches[1][0] < self.margin:
                    continue
                matches[0][1].observe(sample.vector, sample.seconds, block, nearest >= self.new_threshold,
                                      sample.evidence_seconds)
            else:
                self.candidates.append(VoiceCandidate(sample.vector.copy(), sample.seconds, block, block,
                    borderline=nearest >= self.new_threshold, recent_vectors=[sample.vector.copy()],
                    seed_context_seconds=sample.evidence_seconds))
                self.candidates = self.candidates[-self.max_candidates:]
        self._prune_candidates()

    def _scores(self, vector, speakers):
        if not self.learning:
            return [speaker.similarity(vector) for speaker in speakers]
        return self.feature_bank.scores(vector, speakers,
            reference_speakers=[s for s in speakers if s.ready] if self.eager else None)

    def _register(self, vector, seconds, observations=1):
        vector = unit(vector)
        speaker = (self.store.create(vector, observations, seconds) if self.store is not None
                   else Speaker(f'speaker_{self.next_identity}', vector, observations, seconds))
        self.next_identity += 1
        self.speakers.append(speaker)
        return speaker.id

    def _discovery_result(self, status, identity=None, **details):
        self.decisions.append({'status': status, **details})
        return identity

    def _discover(self, vector, seconds, embedding_seconds=None):
        # Brief usable speech must accumulate rather than be discarded forever.
        if seconds < min(MIN_SPEECH_SECONDS, self.min_new_seconds):
            return self._discovery_result('short_speech')
        nearest = max(self._scores(vector, self.speakers), default=-1.0)
        # Keep the first control immediate, but never create later identities
        # from one outlier: transitions and separation artifacts can be very
        # far from every registered voice, even when no new person arrived.
        if self.eager and not self.speakers:
            identity = self._register(vector, seconds)
            return self._discovery_result('created', identity, speaker_id=identity, provisional=True)
        if not self.speakers and seconds >= self.min_new_seconds:
            identity = self._register(vector, seconds)
            return self._discovery_result('created', identity)
        nearest = max(self._scores(vector, self.speakers), default=-1.0)
        # Above this ceiling the voice may be an existing speaker with an
        # ambiguous assignment. Repetition alone must not duplicate that person.
        if nearest >= self.discovery_ceiling:
            return self._discovery_result('ambiguous_existing', best_similarity=nearest)
        borderline = nearest >= self.new_threshold
        available = [(float(vector @ c.vector), c) for c in self.candidates
                     if c.last_block != self.block]
        available.sort(key=lambda item: item[0], reverse=True)
        if (available and available[0][0] >= self.candidate_threshold
                and (len(available) == 1 or available[0][0] - available[1][0] >= self.margin)):
            _, candidate = available[0]
            candidate.observe(vector, seconds, self.block, borderline, embedding_seconds)
            nearest = max(self._scores(candidate.vector, self.speakers), default=-1.0)
            candidate.borderline |= borderline or nearest >= self.new_threshold
            if nearest >= self.discovery_ceiling:
                self.candidates.remove(candidate)
                return self._discovery_result('ambiguous_existing', best_similarity=nearest)
        else:
            # Two separator outputs within one block are not independent evidence.
            if any(float(vector @ c.vector) >= self.candidate_threshold for c in self.candidates):
                return self._discovery_result('ambiguous_candidate')
            candidate = VoiceCandidate(vector, seconds, self.block, self.block, borderline=borderline,
                recent_vectors=[vector.copy()], seed_context_seconds=embedding_seconds)
            self.candidates.append(candidate)
            self.candidates = self.candidates[-self.max_candidates:]
        needed = 2 if self.eager else 3 if candidate.borderline else 2
        seconds_needed = (0.6 if self.eager else
                          self.min_new_seconds if not self.speakers else needed * self.min_new_seconds)
        observations = (candidate.observations if self.eager else
                        candidate.consistent_observations if candidate.borderline else candidate.observations)
        stable_seed = (not self.eager or candidate.seed_seconds >= .75
                       or candidate.consistent_observations >= 3)
        if observations >= needed and candidate.seconds + 1e-8 >= seconds_needed and stable_seed:
            # A brief transition must not remain the person's lifelong anchor.
            # Use the most complete observed segment after cross-window agreement.
            seed = candidate.seed_vector if self.eager else candidate.vector
            identity = self._register(seed, candidate.seconds, candidate.observations)
            self.candidates.remove(candidate)
            return self._discovery_result('created', identity, speaker_id=identity, seed_seconds=candidate.seed_seconds)
        return self._discovery_result('confirming', observations=observations,
                                      required_observations=needed, seconds=candidate.seconds,
                                      required_seconds=seconds_needed, borderline=candidate.borderline,
                                      collected_observations=candidate.observations, needs_stable_seed=not stable_seed)

    def assign(self, vectors, durations, *, reference_eligible=True, embedding_durations=None):
        # Separator output order is arbitrary. Once enrollment has started,
        # examine the more complete current track first; otherwise whichever
        # source happens to be output #0 can supply the next person's anchor.
        # Remap both identities and diagnostics to the caller's original order.
        quality = durations if embedding_durations is None else embedding_durations
        if (self.eager and self.speakers and len(vectors) == len(durations) == len(quality) == 2
                and all(np.isfinite(t) for t in [*durations, *quality])):
            order = sorted(range(2), key=lambda i: (-durations[i], -quality[i], tuple(unit(vectors[i]))))
            result = self._assign([vectors[i] for i in order], [durations[i] for i in order],
                reference_eligible=reference_eligible, embedding_durations=[quality[i] for i in order])
            restored, decisions = [None, None], [None, None]
            for i, identity, decision in zip(order, result, self.decisions):
                restored[i], decisions[i] = identity, decision
            self.decisions = decisions
            return restored
        return self._assign(vectors, durations, reference_eligible=reference_eligible,
                            embedding_durations=embedding_durations)

    def _assign(self, vectors, durations, *, reference_eligible=True, embedding_durations=None):
        if len(vectors) != len(durations) or len(vectors) > 2:
            raise ValueError("每块最多两路声音，特征和时长必须对应")
        embedding_durations = durations if embedding_durations is None else embedding_durations
        if (len(embedding_durations) != len(durations) or any(
                not np.isfinite(q) or not np.isfinite(t) or q + 1e-8 < t or t < 0
                for q, t in zip(embedding_durations, durations))):
            raise ValueError('声纹上下文时长必须有效且不短于本段有效语音')
        self.block += 1
        self.decisions = []
        self.last_merges = []
        self._prune_candidates()
        vectors = [unit(v) for v in vectors]
        known = list(self.speakers)
        scores = np.array([self._scores(v, known) for v in vectors])
        options = []
        for i, row in enumerate(scores):
            allowed = [None]
            # Short utterances have noisier embeddings. A lower match threshold
            # does not lower the separate, multi-block new-identity requirement.
            threshold = self.short_threshold if durations[i] < 2.0 else self.threshold
            for j, score in enumerate(row):
                others = [x for k, x in enumerate(row) if k != j]
                others.extend(float(vectors[i] @ c.vector) for c in self.candidates if not c.borderline)
                if score >= threshold and (not others or score - max(others) >= self.margin):
                    allowed.append(j)
            options.append(allowed)
        best, best_score = tuple([None] * len(vectors)), -1.0
        for choice in product(*options):
            used = [x for x in choice if x is not None]
            if len(set(used)) != len(used):
                # A fixed two-source separator can split ONE voice into two
                # tracks. Reuse the identity only when both independently
                # match it and their own voice features also agree.
                if not self.eager or float(vectors[0] @ vectors[1]) < .35:
                    continue
            score = sum(float(scores[i, j]) for i, j in enumerate(choice) if j is not None)
            if score > best_score:
                best, best_score = choice, score
        result = []
        # For a shared identity only the longer track contributes a learning
        # observation. Duplicate separator outputs are not independent votes.
        primary = {choice: max((i for i, other in enumerate(best) if other == choice),
                               key=lambda i: durations[i]) for choice in best if choice is not None}
        for i, choice in enumerate(best):
            if choice is not None:
                speaker = known[choice]
                competitors = [float(x) for j, x in enumerate(scores[i]) if j != choice]
                competitors.extend(float(vectors[i] @ c.vector) for c in self.candidates if not c.borderline)
                learn = (self.learning and reference_eligible and embedding_durations[i] >= 1.0 and scores[i, choice] >= 0.55
                         and (not competitors or scores[i, choice] - max(competitors) >= 0.15))
                if primary[choice] == i:
                    if self.store is not None:
                        self.store.record_match(speaker, vectors[i], durations[i], learn)
                    else:
                        speaker.observe(vectors[i], durations[i], learn)
                self.decisions.append({'status': 'recognized', 'speaker_id': speaker.id})
                # A reliable return to a known identity resolves its temporary
                # drift candidates; those must not later become duplicates.
                self.candidates = [c for c in self.candidates
                                   if float(vectors[i] @ c.vector) < self.candidate_threshold]
                result.append(speaker.id)
                continue
            # The first output can create an identity in this very block.
            # Let a duplicate second output reuse it without a second profile.
            if (self.eager and i and result[0] is not None
                    and float(vectors[i] @ vectors[0]) >= .35
                    and result[0] not in {s.id for s in known}):
                result.append(result[0])
                self.decisions.append({'status': 'recognized', 'speaker_id': result[0]})
            else:
                result.append(self._discover(vectors[i], durations[i], embedding_durations[i]))
        entries = [(v, seconds, identity, embedding_durations[i]) for i, (v, seconds, identity) in enumerate(zip(vectors, durations, result))
                   if seconds >= MIN_SPEECH_SECONDS and (identity is None or i == max(
                       (j for j, other in enumerate(result) if other == identity), key=lambda j: durations[j]))]
        if self.store is not None and entries:
            self.store.add_samples(entries, reference_eligible=reference_eligible)
            self.feature_bank = FeatureBank(self.store.load_samples())
        else:
            for vector, seconds, identity, evidence_seconds in entries:
                self.feature_bank.add(vector, seconds, identity, self.block, reference_eligible, evidence_seconds)
        if (len(result) == 2 and all(result) and result[0] != result[1]
                and (not self.eager or float(vectors[0] @ vectors[1]) < .20)):
            # Only clearly distinct voices create a lasting cannot-link edge;
            # separator output count alone is not evidence of two humans.
            pair = tuple(sorted(result))
            self.conflicts.add(pair)
            if self.store is not None:
                self.store.record_conflict(*pair)
        before = list((sample.id, sample.speaker_id) for sample in self.feature_bank.samples)
        # A few repeated short matches can light the confirmation indicator,
        # but are still insufficient independent evidence to anchor a class.
        support = {s.id: sum(sample.speaker_id == s.id and sample.evidence_seconds >= 1 and sample.reference_eligible
                           and float(sample.vector @ s.vector) >= .55
                           for sample in self.feature_bank.samples) for s in self.speakers}
        provisional = {s.id for s in self.speakers if not s.ready or support[s.id] < 3} if self.eager else set()
        self.last_corrections = self.feature_bank.reconcile(
            self.speakers, provisional_ids=provisional, conflicts=self.conflicts) if self.learning else []
        self.feature_bank.trim()
        if self.store is not None and (entries or self.last_corrections):
            self.store.correct_samples(self.last_corrections)
            self.feature_bank = FeatureBank(self.store.load_samples())
        if self.eager and self.learning:
            self._merge_corrected_profiles(before, provisional)
            result = [self.aliases.get(identity, identity) for identity in result]
        self._prune_candidates()
        return result

    def _merge_corrected_profiles(self, before, provisional):
        """Retire a draft only when ALL retained samples now agree on one person."""
        current = {sample.id: sample.speaker_id for sample in self.feature_bank.samples}
        for source in sorted(provisional):
            sample_ids = [number for number, identity in before if identity == source]
            destinations = {current.get(number) for number in sample_ids}
            if not sample_ids or len(destinations) != 1:
                continue
            target = destinations.pop()
            if target is None or target == source or tuple(sorted((source, target))) in self.conflicts:
                continue
            if self.store is not None and not self.store.merge(source, target):
                continue
            if self.store is not None:
                self.speakers = self.store.load_speakers()
                self.aliases = self.store.load_aliases()
                self.conflicts = self.store.load_conflicts()
            else:
                self.speakers = [s for s in self.speakers if s.id != source]
                self.aliases = {key: target if value == source else value for key, value in self.aliases.items()}
                self.aliases[source] = target
                self.conflicts = {tuple(sorted(target if item == source else item for item in pair))
                                  for pair in self.conflicts}
            self.last_merges.append({'from': source, 'to': target})

    def _prune_candidates(self):
        # Discovery requires distance from every known voice. After promotion or
        # learning, old candidates that no longer meet that condition must not
        # compete with their own registered identity on subsequent blocks.
        self.candidates = [c for c in self.candidates
                           if self.block - c.last_block <= self.candidate_blocks
                           and self.block - c.first_block <= self.max_candidate_age
                           and max(self._scores(c.vector, self.speakers), default=-1.0) < self.discovery_ceiling]

    @property
    def max_candidate_age(self):
        # Ten 0.2-second utterances need more than six analysis blocks. Retain
        # active candidates, but still bound their total lifetime and memory.
        return max(self.candidate_blocks, 30)

    def snapshot(self):
        if self.store is not None:
            return self.store.summaries()
        return [{"id": s.id, "ready": s.ready, "observations": s.observations,
                 "seconds": round(s.seconds, 2), "saved": self.store is not None,
                 "learned_observations": s.learned_observations,
                 "feature_samples": sum(row.speaker_id == s.id for row in self.feature_bank.samples)}
                for s in self.speakers]


UNCLASSIFIED_ID = '__unclassified__'


@dataclass(frozen=True)
class TrackSelection:
    """Immutable snapshot read by the audio callback, with independent overrides."""
    default_enabled: bool = True
    overrides: tuple = ()
    aliases: tuple = ()

    def enabled(self, identity):
        identity = dict(self.aliases).get(identity, identity)
        return dict(self.overrides).get(identity, self.default_enabled)


@dataclass
class AudioPacket:
    original: np.ndarray
    tracks: dict = field(default_factory=dict)
    captured_at: float = 0.0

    def select(self, mode):
        return self.slice(mode, 0, len(self.original))

    def active(self, start, end):
        """Activity on identified tracks, independent of listening/mute choices."""
        if end <= start:
            return set()
        return {identity for identity, track in self.tracks.items()
                if float(np.mean(track[start:end] ** 2)) > 0.002 ** 2}

    def slice(self, mode, start, end):
        """Read only the callback-sized region, including for mute/missing IDs."""
        if isinstance(mode, TrackSelection):
            include_unclassified = mode.enabled(UNCLASSIFIED_ID)
            result = self.original[start:end].copy() if include_unclassified else np.zeros(end - start, np.float32)
            for identity, track in self.tracks.items():
                if include_unclassified and not mode.enabled(identity):
                    result -= track[start:end]
                elif not include_unclassified and mode.enabled(identity):
                    result += track[start:end]
            # The remainder includes unrecognized speech and separation errors,
            # not a clean background stem. All-on is the exact original signal.
            return result
        if mode.startswith('exclude:'):
            excluded = mode.split(':', 1)[1]
            result = np.zeros(end - start, np.float32)
            # Only pass identified, separated tracks. Never subtract from the mixture:
            # that would leak the excluded person when identity matching fails.
            for identity, track in self.tracks.items():
                if identity != excluded:
                    result += track[start:end]
            return result
        track = self.original if mode == "all" else self.tracks.get(mode)
        return track[start:end] if track is not None else np.zeros(end - start, np.float32)


class OutputControl:
    def __init__(self):
        self._lock = Lock()
        self.mode = "all"
        self.gain = 0.4
        self.aliases = {}

    def set(self, mode=None, gain=None):
        with self._lock:
            if mode is not None:
                self.mode = mode
            if gain is not None:
                self.gain = float(np.clip(gain, 0, 1))

    def get(self):
        with self._lock:
            mode = self.mode
            if isinstance(mode, TrackSelection):
                mode = TrackSelection(mode.default_enabled, mode.overrides, tuple(self.aliases.items()))
            return mode, self.gain

    def canonical(self, identity):
        with self._lock:
            return self.aliases.get(identity, identity)

    def apply_aliases(self, aliases):
        with self._lock:
            for source, target in aliases.items():
                source = self.aliases.get(source, source)
                target = self.aliases.get(target, target)
                if source == target:
                    continue
                selection = self.mode if isinstance(self.mode, TrackSelection) else TrackSelection(self.mode == 'all')
                enabled = selection.enabled(source) and selection.enabled(target)
                overrides = dict(selection.overrides)
                overrides.pop(source, None)
                overrides.pop(target, None)
                if enabled != selection.default_enabled:
                    overrides[target] = enabled
                self.aliases = {key: target if value == source else value for key, value in self.aliases.items()}
                self.aliases[source] = target
                self.mode = (TrackSelection(selection.default_enabled, tuple(overrides.items())) if overrides
                             else 'all' if selection.default_enabled else 'mute')

    def set_track(self, identity, enabled):
        with self._lock:
            identity = self.aliases.get(identity, identity)
            selection = self.mode if isinstance(self.mode, TrackSelection) else TrackSelection(
                default_enabled=self.mode == 'all')
            overrides = dict(selection.overrides)
            if bool(enabled) == selection.default_enabled:
                overrides.pop(identity, None)
            else:
                overrides[identity] = bool(enabled)
            # Undoing the final exclusion must restore the original signal,
            # including blocks that identity matching could not label. Keep
            # filtering while any exclusion or default-mute policy remains.
            if not overrides:
                self.mode = 'all' if selection.default_enabled else 'mute'
            else:
                self.mode = TrackSelection(selection.default_enabled, tuple(overrides.items()))

    def track_enabled(self, identity):
        with self._lock:
            identity = self.aliases.get(identity, identity)
            if isinstance(self.mode, TrackSelection):
                return self.mode.enabled(identity)
            return self.mode == 'all'


class SpeakingActivity:
    """Short hold bridges syllable gaps; snapshots expire even if callbacks stop."""
    def __init__(self, hold_seconds=0.3):
        self.hold_seconds = hold_seconds
        self.lock = Lock()
        self.seen = {}

    def observe(self, identities, now):
        with self.lock:
            self.seen = {key: stamp for key, stamp in self.seen.items()
                         if now - stamp < self.hold_seconds}
            self.seen.update({key: now for key in identities})

    def active(self, now=None):
        now = time.perf_counter() if now is None else now
        with self.lock:
            return {key for key, stamp in self.seen.items() if now - stamp < self.hold_seconds}


class PlaybackBuffer:
    """Bounded, callback-safe packet queue. Selection applies at output time."""
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.lock = Lock()
        self.packets = []
        self.offset = 0
        self.samples = 0
        self.dropped = 0
        self.underruns = 0
        self.last_age = 0.0
        self.activity = SpeakingActivity()

    def put(self, packet):
        with self.lock:
            self.packets.append(packet)
            self.samples += len(packet.original)
            while self.samples > self.max_samples and len(self.packets) > 1:
                oldest = self.packets.pop(0)
                self.samples -= len(oldest.original) - self.offset
                self.offset = 0
                self.dropped += 1

    def read(self, frames, mode, gain, now):
        output = np.zeros(frames, np.float32)
        active = set()
        with self.lock:
            written = 0
            while self.packets and written < frames:
                packet = self.packets[0]
                count = min(frames - written, len(packet.original) - self.offset)
                output[written:written + count] = packet.slice(mode, self.offset, self.offset + count)
                active.update(packet.active(self.offset, self.offset + count))
                if packet.captured_at:
                    self.last_age = max(0.0, now - packet.captured_at)
                self.offset += count
                written += count
                self.samples -= count
                if self.offset == len(packet.original):
                    self.packets.pop(0)
                    self.offset = 0
            if written < frames:
                self.underruns += 1
        self.activity.observe(active, now)
        return np.clip(output * gain, -0.95, 0.95)
