"""Model-independent identity and playback rules. No audio device access."""
from dataclasses import dataclass, field
from itertools import product
from threading import Lock

import numpy as np
from scipy.signal import resample_poly
from math import gcd


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
    """Session-only identities; each separator block still has at most two sources.

    Joint one-to-one assignment handles separator output permutations.
    The first voice is immediately selectable. Later identities need consistent
    evidence across separate blocks, not a single mismatched short utterance.
    """
    def __init__(self, threshold=0.55, new_threshold=0.30, margin=0.10, min_new_seconds=1.0,
                 short_threshold=0.45, candidate_threshold=0.50,
                 candidate_blocks=6, max_candidates=12):
        self.threshold = threshold
        self.short_threshold = short_threshold
        self.new_threshold = new_threshold
        self.margin = margin
        self.min_new_seconds = min_new_seconds
        self.candidate_threshold = candidate_threshold
        self.candidate_blocks = candidate_blocks
        self.max_candidates = max_candidates
        self.speakers = []
        self.candidates = []
        self.block = 0

    def _discover(self, vector, seconds):
        if seconds < self.min_new_seconds:
            return None
        if not self.speakers:
            speaker = Speaker('speaker_1', vector, seconds=seconds)
            self.speakers.append(speaker)
            return speaker.id
        # A poor match is not evidence of a new person. Keep a wide uncertainty
        # band between recognition and discovery, including for short windows.
        if max(float(vector @ s.vector) for s in self.speakers) >= self.new_threshold:
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
            if max(float(candidate.vector @ s.vector) for s in self.speakers) >= self.new_threshold:
                self.candidates.remove(candidate)
                return None
            if candidate.observations >= 2 and candidate.seconds >= 2 * self.min_new_seconds:
                speaker = Speaker(f'speaker_{len(self.speakers) + 1}', candidate.vector,
                                  observations=candidate.observations, seconds=candidate.seconds)
                self.speakers.append(speaker)
                self.candidates.remove(candidate)
                return speaker.id
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
        self.candidates = [c for c in self.candidates if self.block - c.first_block <= self.candidate_blocks]
        vectors = [unit(v) for v in vectors]
        known = list(self.speakers)
        scores = np.array([[float(v @ s.vector) for s in known] for v in vectors])
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
                speaker.observations += 1
                speaker.seconds += durations[i]
                result.append(speaker.id)
                continue
            result.append(self._discover(vectors[i], durations[i]))
        return result

    def snapshot(self):
        return [{"id": s.id, "ready": s.ready, "observations": s.observations,
                 "seconds": round(s.seconds, 2)} for s in self.speakers]


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
            overrides[identity] = bool(enabled)
            self.mode = TrackSelection(selection.default_enabled, tuple(overrides.items()))

    def track_enabled(self, identity):
        with self._lock:
            if isinstance(self.mode, TrackSelection):
                return self.mode.enabled(identity)
            return self.mode == 'all'


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
        with self.lock:
            written = 0
            while self.packets and written < frames:
                packet = self.packets[0]
                count = min(frames - written, len(packet.original) - self.offset)
                output[written:written + count] = packet.slice(mode, self.offset, self.offset + count)
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
        return np.clip(output * gain, -0.95, 0.95)
