"""Bounded local feature memory and leave-one-out nearest-neighbor correction.

No waveform storage. Speaker IDs remain stable when individual samples move.
"""
from dataclasses import dataclass

import numpy as np


MAX_SAMPLES = 512
MAX_PER_SPEAKER = 64
MAX_PENDING = 128
MIN_SPEECH_SECONDS = 0.2


@dataclass
class VoiceSample:
    id: int
    vector: np.ndarray
    seconds: float
    speaker_id: str | None
    batch_id: int = 0


class FeatureBank:
    def __init__(self, samples=()):
        self.samples = list(samples)
        self.next_id = max((s.id for s in self.samples), default=0) + 1

    def add(self, vector, seconds, identity, batch_id=0):
        sample = VoiceSample(self.next_id, vector.copy(), seconds, identity, batch_id)
        self.next_id += 1
        self.samples.append(sample)
        return sample

    def trim(self):
        kept, counts = [], {}
        for sample in reversed(self.samples):
            key = sample.speaker_id
            limit = MAX_PENDING if key is None else MAX_PER_SPEAKER
            if counts.get(key, 0) < limit and len(kept) < MAX_SAMPLES:
                kept.append(sample)
                counts[key] = counts.get(key, 0) + 1
        self.samples = list(reversed(kept))

    @staticmethod
    def reference_samples(samples, speakers):
        """Predicted labels alone are not evidence that a sample is reliable.

        Use only sufficiently long samples close to their ORIGINAL anchor,
        with a clear lead over other anchors. Never let a weak prediction vote
        itself into stronger future predictions through the learned centroid.
        """
        accepted = []
        for sample in samples:
            if sample.speaker_id is None or sample.seconds < 1.0:
                continue
            scores = {speaker.id: float(sample.vector @ speaker.vector) for speaker in speakers}
            own = scores.get(sample.speaker_id, -1.0)
            others = [score for identity, score in scores.items() if identity != sample.speaker_id]
            if own >= .55 and (not others or own - max(others) >= .15):
                accepted.append(sample)
        return accepted

    def scores(self, vector, speakers, exclude_id=None, samples=None):
        samples = self.samples if samples is None else samples
        samples = self.reference_samples(samples, speakers)
        result = []
        for speaker in speakers:
            neighbors = sorted((float(vector @ s.vector) for s in samples
                                if s.speaker_id == speaker.id and s.id != exclude_id), reverse=True)
            # Each person gets equal weight regardless of how many samples they
            # have. A single outlier cannot replace an established voice model.
            reference = speaker.similarity(vector)
            result.append(max(reference, float(np.mean(neighbors[:3]))) if len(neighbors) >= 3
                          else reference)
        return result

    def reconcile(self, speakers):
        """Simultaneous corrections; no self-votes or same-pass cascades.

        Three supporting neighbors, high absolute similarity and a clear lead
        are required. Sparse/ambiguous samples stay pending or keep their label.
        """
        snapshot = list(self.samples)
        if not snapshot or not speakers:
            return []
        # One matrix multiplication replaces hundreds of thousands of tiny
        # Python dot products when the feature memory is full.
        vectors = np.stack([sample.vector for sample in snapshot])
        similarities = vectors @ vectors.T
        reference_ids = {sample.id for sample in self.reference_samples(snapshot, speakers)}
        groups = {speaker.id: np.array([i for i, sample in enumerate(snapshot)
                                        if sample.speaker_id == speaker.id and sample.id in reference_ids], dtype=int) for speaker in speakers}
        changes = []
        for index, sample in enumerate(snapshot):
            votes, scores = [], {}
            for speaker in speakers:
                indices = groups[speaker.id]
                neighbors = similarities[index, indices[indices != index]]
                reference = speaker.similarity(sample.vector)
                scores[speaker.id] = reference
                if len(neighbors) >= 3:
                    score = float(np.mean(np.partition(neighbors, -3)[-3:]))
                    votes.append((score, speaker.id))
                    scores[speaker.id] = max(reference, score)
            votes.sort(reverse=True)
            if not votes or votes[0][0] < .75:
                continue
            score, winner = votes[0]
            if winner == sample.speaker_id:
                continue
            if sample.batch_id and any(other.id != sample.id and other.batch_id == sample.batch_id
                                       and other.speaker_id == winner for other in snapshot):
                continue  # Two simultaneous sources cannot be the same person.
            # Include sparse classes' reference scores as competing evidence.
            competing = [score for identity, score in scores.items() if identity != winner]
            if competing and score - max(competing) < .15:
                continue
            changes.append((sample.id, sample.speaker_id, winner))
        groups = {s.id: s.batch_id for s in snapshot}
        changes = [change for change in changes if not groups[change[0]] or sum(
            groups[other[0]] == groups[change[0]] and other[2] == change[2] for other in changes) == 1]
        assignments = {identity: new for identity, _, new in changes}
        for sample in self.samples:
            if sample.id in assignments:
                sample.speaker_id = assignments[sample.id]
        return changes
