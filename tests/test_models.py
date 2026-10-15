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
        seconds = .4
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
    assert '0.40 秒' in short['recognition_status'] and not store.load_samples()
    backend.seconds = 1.2
    _, first = processor.process(audio)
    assert '已新增 1 人' in first['recognition_status']
    backend.vector = np.eye(192, dtype=np.float32)[1]
    backend.seconds = .5
    _, pending = processor.process(audio)
    assert '1/2' in pending['recognition_status'] and '0.5/2.0 秒' in pending['recognition_status']
    assert pending['feature_samples'] == 2 and len(store.load_speakers()) == 1
    processor.process(audio)
    _, not_yet = processor.process(audio)
    assert len(store.load_speakers()) == 1 and not_yet['pending_speakers'] == 1
    _, added = processor.process(audio)
    assert '已新增 1 人' in added['recognition_status'] and len(store.load_speakers()) == 2
    assert len(ProfileStore(store.path).load_samples()) == 5
