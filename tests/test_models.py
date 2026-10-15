import numpy as np
import pytest
from selective_hearing.models import calibrate_sources, Processor
from selective_hearing.core import SpeakerRegistry
from selective_hearing.profiles import ProfileStore


def test_scale_ambiguous_predictions_are_calibrated_without_clean_input():
    rng = np.random.default_rng(47)
    a, b = rng.normal(0, .08, (2, 16000))
    estimate = np.array([a * 60000, b * -40000])
    corrected = calibrate_sources(estimate, a + b)
    np.testing.assert_allclose(corrected[0], a, atol=1e-7)
    np.testing.assert_allclose(corrected[1], b, atol=1e-7)
    assert np.max(np.abs(corrected)) < .95


def test_zero_predictions_stay_silent_and_invalid_predictions_fail():
    assert np.all(calibrate_sources(np.zeros((2, 30)), np.zeros(30)) == 0)
    with pytest.raises(ValueError):
        calibrate_sources(np.full((2, 30), np.nan), np.zeros(30))


def test_discovery_status_explains_weak_short_and_accumulating_speech(tmp_path):
    class Backend:
        seconds = .1
        vector = np.eye(192, dtype=np.float32)[0]

        def separate(self, audio):
            return [audio]

        def speech(self, audio):
            return audio[:int(self.seconds * 16000)]

        def embedding(self, audio):
            return self.vector

    store = ProfileStore(tmp_path / 'profiles.sqlite3')
    backend = Backend()
    processor = Processor(backend, SpeakerRegistry(store=store))
    audio = np.full(24000, .03, np.float32)
    _, weak = processor.process(audio * .001)
    assert '输入过弱' in weak['recognition_status']
    _, short = processor.process(audio)
    assert '0.10 秒' in short['recognition_status'] and not store.load_samples()
    backend.seconds = 1.2
    _, first = processor.process(audio)
    assert '已新增 1 人' in first['recognition_status']
    backend.vector = np.eye(192, dtype=np.float32)[1]
    backend.seconds = .5
    _, pending = processor.process(audio)
    assert '已新增 1 人' in pending['recognition_status'] and pending['active'] == ['speaker_2']
    assert pending['feature_samples'] == 2 and len(store.load_speakers()) == 2
    processor.process(audio)
    _, not_yet = processor.process(audio)
    assert len(store.load_speakers()) == 2 and not_yet['pending_speakers'] == 0
    _, added = processor.process(audio)
    assert '匹配到已有 1 人' in added['recognition_status'] and len(store.load_speakers()) == 2
    assert len(ProfileStore(store.path).load_samples()) == 5


def test_short_known_voice_is_matched_and_muted_without_training(tmp_path):
    from selective_hearing.core import OutputControl
    class Backend:
        def separate(self, audio): return [audio]
        def speech(self, audio): return audio[:4800]
        def embedding(self, audio): return np.eye(192, dtype=np.float32)[0]
    store = ProfileStore(tmp_path / 'profiles.sqlite3')
    person = store.create(np.eye(192, dtype=np.float32)[0], 1, 1.2)
    processor = Processor(Backend(), SpeakerRegistry(store=store))
    # Also covers the final short file block, formerly skipped below one second.
    packet, report = processor.process(np.full(4800, .03, np.float32))
    assert report['active'] == [person.id] and report['short_speech_tracks'] == 1
    assert store.load_speakers()[0].learned_observations == 0
    control = OutputControl()
    control.set_track(person.id, False)  # unclassified remains on
    np.testing.assert_allclose(packet.select(control.get()[0]), 0, atol=1e-8)


def test_short_new_voices_are_created_immediately_and_survive_restart(tmp_path):
    class Backend:
        voice = 0
        def separate(self, audio): return [audio]
        def speech(self, audio): return audio[:4000]
        def embedding(self, audio): return np.eye(192, dtype=np.float32)[self.voice]
    path = tmp_path / 'profiles.sqlite3'
    backend = Backend()
    processor = Processor(backend, SpeakerRegistry(store=ProfileStore(path)))
    audio = np.full(24000, .03, np.float32)
    for index in range(4):
        packet, report = processor.process(audio)
        assert len(processor.registry.speakers) == 1
    assert report['active'] == ['speaker_1']
    backend.voice = 1
    for index in range(8):
        if index == 5:
            processor = Processor(backend, SpeakerRegistry(store=ProfileStore(path)))
        packet, report = processor.process(audio)
        assert len(processor.registry.speakers) == 2
    assert report['active'] == ['speaker_2']
    assert len(ProfileStore(path).load_samples()) == 12
