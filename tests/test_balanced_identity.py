import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry, OutputControl
from selective_hearing.models import Processor
from selective_hearing.profiles import ProfileStore


def vector(axis=0):
    return np.eye(192, dtype=np.float32)[axis]


def variant(similarity, axis=1):
    return vector() * similarity + vector(axis) * np.sqrt(1 - similarity ** 2)


@pytest.mark.parametrize('persistent', [False, True])
def test_single_voice_variation_does_not_generate_people_or_poison_learning(tmp_path, persistent):
    store = ProfileStore(tmp_path / 'voices.sqlite3') if persistent else None
    registry = SpeakerRegistry(store=store)
    assert registry.assign([vector()], [.3]) == ['speaker_1']
    for i in range(100):
        # Legitimate short-frame variation, below the previous .45 match gate.
        assert registry.assign([variant(.4, axis=1 + i % 190)], [.3]) == ['speaker_1']
    assert len(registry.speakers) == 1
    assert registry.speakers[0].learned_observations == 0
    assert len(registry.feature_bank.samples) <= 64


def test_ambiguous_short_outliers_stay_bounded_instead_of_becoming_people():
    registry = SpeakerRegistry()
    registry.assign([vector()], [.3])
    for i in range(100):
        assert registry.assign([variant(.25, axis=1 + i)], [.3]) == [None]
    assert len(registry.speakers) == 1
    assert len(registry.candidates) <= registry.max_candidates
    for _ in range(7):
        registry.assign([], [])
    assert not registry.candidates


def test_new_people_are_not_capped_at_two_and_restore_from_disk(tmp_path):
    path = tmp_path / 'voices.sqlite3'
    registry = SpeakerRegistry(store=ProfileStore(path))
    for i in range(8):
        assert registry.assign([vector(i)], [1.2]) == [f'speaker_{i + 1}']
    restored = SpeakerRegistry(store=ProfileStore(path))
    for i in reversed(range(8)):
        assert restored.assign([vector(i)], [.25]) == [f'speaker_{i + 1}']
    assert len(restored.speakers) == 8


def test_duplicate_separator_outputs_count_once_and_share_playback_controls(tmp_path):
    class Backend:
        def separate(self, audio): return [audio * .4, audio * .6]
        def speech(self, audio): return audio
        def embedding(self, audio): return vector()
    store = ProfileStore(tmp_path / 'voices.sqlite3')
    processor = Processor(Backend(), SpeakerRegistry(store=store))
    audio = np.full(16000, .05, np.float32)
    for _ in range(4):
        packet, report = processor.process(audio)
    assert len(store.load_speakers()) == 1 and not store.load_conflicts()
    assert store.load_speakers()[0].observations == 4
    assert len(store.load_samples()) == 4
    assert report['active'] == ['speaker_1']
    assert '匹配到已有 1 人' in report['recognition_status']
    np.testing.assert_allclose(packet.tracks['speaker_1'], audio)
    control = OutputControl()
    control.set_track('speaker_1', False)
    np.testing.assert_allclose(packet.select(control.get()[0]), 0, atol=1e-7)
    control.set_track('speaker_1', True)
    np.testing.assert_array_equal(packet.select(control.get()[0]), audio)


def test_shared_identity_uses_longer_track_for_single_learning_observation(tmp_path):
    store = ProfileStore(tmp_path / 'voices.sqlite3')
    registry = SpeakerRegistry(store=store)
    registry.assign([vector()], [1.2])
    assert registry.assign([vector(), vector()], [.3, 1.2]) == ['speaker_1', 'speaker_1']
    person = store.load_speakers()[0]
    assert person.observations == 2 and person.seconds == pytest.approx(2.4)
    assert person.learned_observations == 1
    assert [sample.seconds for sample in store.load_samples()] == [1.2, 1.2]


def test_distinct_simultaneous_voices_keep_ids_across_output_permutations(tmp_path):
    store = ProfileStore(tmp_path / 'voices.sqlite3')
    registry = SpeakerRegistry(store=store)
    assert registry.assign([vector(0), vector(1)], [1.2, 1.2]) == ['speaker_1', 'speaker_2']
    for _ in range(5):
        assert registry.assign([vector(1), vector(0)], [.3, .3]) == ['speaker_2', 'speaker_1']
    assert store.load_conflicts() == {('speaker_1', 'speaker_2')}
    assert len(store.load_speakers()) == 2


def test_two_outputs_cannot_promote_pending_voice_within_same_window():
    registry = SpeakerRegistry()
    registry.assign([vector(0)], [.3])
    assert registry.assign([vector(1), vector(1)], [.4, .4]) == [None, None]
    assert len(registry.speakers) == 1 and registry.candidates[0].observations == 1
    assert registry.assign([vector(1)], [.4]) == ['speaker_2']


def test_candidates_do_not_resurrect_after_idle_expiry_on_restart(tmp_path):
    path = tmp_path / 'voices.sqlite3'
    registry = SpeakerRegistry(store=ProfileStore(path))
    registry.assign([vector()], [.3])
    registry.assign([vector(1)], [.3])
    # Persist newer blocks using the existing speaker.
    for _ in range(32):
        registry.assign([vector()], [.3])
    again = SpeakerRegistry(store=ProfileStore(path))
    assert not again.candidates and len(again.speakers) == 1
