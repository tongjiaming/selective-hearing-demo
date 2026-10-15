import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry
from selective_hearing.feature_bank import FeatureBank
from selective_hearing.models import Processor
from selective_hearing.profiles import ProfileStore


class ContextBackend:
    def __init__(self):
        self.inputs = []
        self.embedding_lengths = []

    def separate(self, audio):
        self.inputs.append(audio.copy())
        return [audio]

    def speech_mask(self, audio):
        return np.abs(audio) > .01

    def speech(self, audio):
        return audio[self.speech_mask(audio)]

    def embedding(self, speech):
        self.embedding_lengths.append(len(speech))
        return np.eye(192, dtype=np.float32)[0]


def test_past_context_does_not_replay_audio_or_double_count_speech(tmp_path):
    backend = ContextBackend()
    store = ProfileStore(tmp_path / 'voices.sqlite3')
    processor = Processor(backend, SpeakerRegistry(store=store))
    first = np.full(24000, .03, np.float32)
    second = np.full(24000, .06, np.float32)
    a, _ = processor.process(first, captured_at=12)
    b, report = processor.process(second, captured_at=13.5)
    np.testing.assert_array_equal(backend.inputs[0], first)
    np.testing.assert_array_equal(backend.inputs[1], np.concatenate([first, second]))
    np.testing.assert_array_equal(a.tracks['speaker_1'], first)
    np.testing.assert_array_equal(b.tracks['speaker_1'], second)
    assert b.captured_at == 13.5 and report['seconds'] == 1.5
    assert backend.embedding_lengths == [24000, 48000]
    assert store.load_speakers()[0].seconds == pytest.approx(3)
    assert [s.seconds for s in store.load_samples()] == [1.5, 1.5]


def test_reused_context_is_saved_but_not_counted_as_independent_learning(tmp_path):
    path = tmp_path / 'voices.sqlite3'
    processor = Processor(ContextBackend(), SpeakerRegistry(store=ProfileStore(path)))
    for _ in range(4):
        processor.process(np.full(24000, .03, np.float32))
    restored = ProfileStore(path)
    samples, people = restored.load_samples(), restored.load_speakers()
    assert [s.reference_eligible for s in samples] == [True, False, True, False]
    assert len(FeatureBank.reference_samples(samples, people)) == 2
    assert people[0].observations == 4 and people[0].learned_observations == 1


def test_old_speech_cannot_keep_person_active_during_current_background():
    processor = Processor(ContextBackend())
    processor.process(np.full(24000, .03, np.float32))
    # Enough energy to enter processing, but current block has no voiced frames.
    packet, report = processor.process(np.full(24000, .005, np.float32))
    assert report['active'] == [] and not packet.tracks
    assert len(processor.registry.feature_bank.samples) == 1
    processor.process(np.zeros(48000, np.float32))
    assert len(processor.history) == 48000 and not processor.history.any()


def test_long_user_window_is_never_truncated_to_context_length():
    backend = ContextBackend()
    processor = Processor(backend)
    for _ in range(2):
        packet, report = processor.process(np.full(64000, .03, np.float32))
        assert len(packet.original) == 64000 and len(packet.tracks['speaker_1']) == 64000
        assert report['seconds'] == 4 and processor.reference_eligible
    assert [len(x) for x in backend.inputs] == [64000, 64000]


def test_new_processor_drops_previous_session_audio_context():
    backend = ContextBackend()
    Processor(backend).process(np.full(24000, .03, np.float32))
    Processor(backend).process(np.full(24000, .06, np.float32))
    assert [len(x) for x in backend.inputs] == [24000, 24000]


def test_disabling_context_reproduces_independent_analysis_windows():
    backend = ContextBackend()
    processor = Processor(backend, context_seconds=0)
    for _ in range(2):
        processor.process(np.full(24000, .03, np.float32))
    assert [len(x) for x in backend.inputs] == [24000, 24000]
    assert not len(processor.history)
