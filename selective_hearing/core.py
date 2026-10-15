"""Model-independent identity and playback rules. No audio device access."""
from dataclasses import dataclass, field
from itertools import product
from threading import Lock

import numpy as np
from scipy.signal import resample_poly
from math import gcd
import time


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
        return max(float(vector @ self.vector),
                   float(vector @ self.learned_vector) if self.learned_vector is not None else -1.0)

    def observe(self, vector, seconds, learn):
        self.observations += 1
        self.seconds += seconds
        # Keep the original reference intact. Learning cannot walk indefinitely
        # toward an unrelated voice through a chain of borderline matches.
        if not learn or float(vector @ self.vector) < 0.35 or self.similarity(vector) < 0.55:
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


class SpeakerRegistry:
    """Optionally persistent identities; each separator block has at most two sources.

    Joint one-to-one assignment handles separator output permutations.
    The first voice is immediately selectable. Later identities need consistent
    evidence across separate blocks, not a single mismatched short utterance.
    """
    def __init__(self, threshold=0.55, new_threshold=0.30, margin=0.10, min_new_seconds=1.0,
                 short_threshold=0.45, candidate_threshold=0.50,
                 candidate_blocks=6, max_candidates=12, store=None, learning=True):
        self.threshold = threshold
        self.short_threshold = short_threshold
        self.new_threshold = new_threshold
        self.margin = margin
        self.min_new_seconds = min_new_seconds
        self.candidate_threshold = candidate_threshold
        self.candidate_blocks = candidate_blocks
        self.max_candidates = max_candidates
        self.store = store
        self.learning = learning
        self.speakers = store.load_speakers() if store is not None else []
        self.candidates = []
        self.block = 0

    def _register(self, vector, seconds, observations=1):
        vector = unit(vector)
        speaker = (self.store.create(vector, observations, seconds) if self.store is not None
                   else Speaker(f'speaker_{len(self.speakers) + 1}', vector, observations, seconds))
        self.speakers.append(speaker)
        return speaker.id

    def _discover(self, vector, seconds):
        if seconds < self.min_new_seconds:
            return None
        if not self.speakers:
            return self._register(vector, seconds)
        # A poor match is not evidence of a new person. Keep a wide uncertainty
        # band between recognition and discovery, including for short windows.
        if max(s.similarity(vector) for s in self.speakers) >= self.new_threshold:
            return None
        available = [(float(vector @ c.vector), c) for c in self.candidates
                     if c.last_block != self.block]
        available.sort(key=lambda item: item[0], reverse=True)
        if (available and available[0][0] >= self.candidate_threshold
                and (len(available) == 1 or available[0][0] - available[1][0] >= self.margin)):
            candidate = available[0][1]
            candidate.vector = unit(candidate.vector * candidate.seconds + vector * seconds)
            candidate.seconds += seconds
            candidate.observations += 1
            candidate.last_block = self.block
            # Recheck the aggregated reference; never promote a candidate that
            # now resembles an existing identity or a just-promoted candidate.
            if max(s.similarity(candidate.vector) for s in self.speakers) >= self.new_threshold:
                self.candidates.remove(candidate)
                return None
            if candidate.observations >= 2 and candidate.seconds >= 2 * self.min_new_seconds:
                identity = self._register(candidate.vector, candidate.seconds, candidate.observations)
                self.candidates.remove(candidate)
                return identity
        else:
            # Duplicate separator outputs in one block cannot count as two
            # sightings, nor occupy multiple slots for the same candidate.
            if any(float(vector @ c.vector) >= self.candidate_threshold for c in self.candidates):
                return None
            self.candidates.append(VoiceCandidate(vector, seconds, self.block, self.block))
            self.candidates = self.candidates[-self.max_candidates:]
        return None

    def assign(self, vectors, durations):
        if len(vectors) != len(durations) or len(vectors) > 2:
            raise ValueError("每块最多两路声音，特征和时长必须对应")
        self.block += 1
        self._prune_candidates()
        vectors = [unit(v) for v in vectors]
        known = list(self.speakers)
        scores = np.array([[s.similarity(v) for s in known] for v in vectors])
        options = []
        for i, row in enumerate(scores):
            allowed = [None]
            # Short utterances have noisier embeddings. A lower match threshold
            # does not lower the separate, multi-block new-identity requirement.
            threshold = self.short_threshold if durations[i] < 2.0 else self.threshold
            for j, score in enumerate(row):
                others = [x for k, x in enumerate(row) if k != j]
                others.extend(float(vectors[i] @ c.vector) for c in self.candidates)
                if score >= threshold and (not others or score - max(others) >= self.margin):
                    allowed.append(j)
            options.append(allowed)
        best, best_score = tuple([None] * len(vectors)), -1.0
        for choice in product(*options):
            used = [x for x in choice if x is not None]
            if len(set(used)) != len(used):
                continue
            score = sum(float(scores[i, j]) for i, j in enumerate(choice) if j is not None)
            if score > best_score:
                best, best_score = choice, score
        result = []
        for i, choice in enumerate(best):
            if choice is not None:
                speaker = known[choice]
                competitors = [float(x) for j, x in enumerate(scores[i]) if j != choice]
                competitors.extend(float(vectors[i] @ c.vector) for c in self.candidates)
                learn = (self.learning and durations[i] >= 1.0 and scores[i, choice] >= 0.55
                         and (not competitors or scores[i, choice] - max(competitors) >= 0.15))
                if self.store is not None:
                    self.store.record_match(speaker, vectors[i], durations[i], learn)
                else:
                    speaker.observe(vectors[i], durations[i], learn)
                result.append(speaker.id)
                continue
            result.append(self._discover(vectors[i], durations[i]))
        self._prune_candidates()
        return result

    def _prune_candidates(self):
        # Discovery requires distance from every known voice. After promotion or
        # learning, old candidates that no longer meet that condition must not
        # compete with their own registered identity on subsequent blocks.
        self.candidates = [c for c in self.candidates
                           if self.block - c.first_block <= self.candidate_blocks
                           and all(s.similarity(c.vector) < self.new_threshold for s in self.speakers)]

    def snapshot(self):
        return [{"id": s.id, "ready": s.ready, "observations": s.observations,
                 "seconds": round(s.seconds, 2), "saved": self.store is not None,
                 "learned_observations": s.learned_observations} for s in self.speakers]


@dataclass(frozen=True)
class TrackSelection:
    """Immutable snapshot read by the audio callback, with independent overrides."""
    default_enabled: bool = True
    overrides: tuple = ()

    def enabled(self, identity):
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
            result = np.zeros(end - start, np.float32)
            for identity, track in self.tracks.items():
                if mode.enabled(identity):
                    result += track[start:end]
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

    def set(self, mode=None, gain=None):
        with self._lock:
            if mode is not None:
                self.mode = mode
            if gain is not None:
                self.gain = float(np.clip(gain, 0, 1))

    def get(self):
        with self._lock:
            return self.mode, self.gain

    def set_track(self, identity, enabled):
        with self._lock:
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
