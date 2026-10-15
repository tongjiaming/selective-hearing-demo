import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry
from selective_hearing.feature_bank import FeatureBank
from selective_hearing.profiles import ProfileStore
from selective_hearing.models import Processor
from test_voice_context import ContextBackend


def voice(axis):
    return np.eye(192, dtype=np.float32)[axis]


def test_single_long_outliers_do_not_create_permanent_people():
    registry = SpeakerRegistry()
    registry.assign([voice(0)], [1.2])
    for axis in range(1, 80):
        assert registry.assign([voice(axis)], [1.5]) == [None]
    assert len(registry.speakers) == 1
    assert len(registry.candidates) <= registry.max_candidates


def test_transition_seed_is_replaced_by_complete_correlated_segment_after_restart(tmp_path):
    path = tmp_path / 'people.sqlite3'
    registry = SpeakerRegistry(store=ProfileStore(path))
    registry.assign([voice(0)], [1.2])
    transition = .6 * voice(1) + .8 * voice(2)
    assert registry.assign([transition], [.3]) == [None]
    registry = SpeakerRegistry(store=ProfileStore(path))
    assert registry.assign([voice(1)], [1.2]) == ['speaker_2']
    np.testing.assert_allclose(registry.speakers[1].vector, voice(1))
    assert registry.speakers[1].seconds == pytest.approx(1.5)
    assert registry.assign([voice(1)], [.25]) == ['speaker_2']


def test_seed_uses_fuller_context_once_current_segment_is_sufficient(tmp_path):
    registry = SpeakerRegistry(store=ProfileStore(tmp_path / 'people.sqlite3'))
    registry.assign([voice(0)], [1.2])
    initial = .6 * voice(1) + .8 * voice(2)
    assert registry.assign([initial], [1.2], embedding_durations=[1.3]) == [None]
    registry = SpeakerRegistry(store=ProfileStore(registry.store.path))
    assert registry.assign([voice(1)], [1.0], embedding_durations=[2.5]) == ['speaker_2']
    np.testing.assert_allclose(registry.speakers[1].vector, voice(1))
    assert registry.speakers[1].seconds == pytest.approx(2.2)


def test_independent_full_context_can_learn_from_short_current_tail(tmp_path):
    store = ProfileStore(tmp_path / 'people.sqlite3')
    processor = Processor(ContextBackend(), SpeakerRegistry(store=store))
    processor.process(np.full(24000, .03, np.float32))
    processor.process(np.full(24000, .04, np.float32))
    tail = np.concatenate([np.full(4800, .05, np.float32), np.zeros(19200, np.float32)])
    processor.process(tail)
    restored = ProfileStore(store.path)
    samples, people = restored.load_samples(), restored.load_speakers()
    assert samples[-1].seconds == pytest.approx(.3)
    assert samples[-1].evidence_seconds == pytest.approx(1.8)
    assert samples[-1].reference_eligible
    assert people[0].learned_observations == 1 and people[0].seconds == pytest.approx(3.3)
    assert len(FeatureBank.reference_samples(samples, people)) == 2


@pytest.mark.parametrize('quality', [None, [.3], [2.0]])
def test_overlapping_context_does_not_supply_learning_votes(quality):
    registry = SpeakerRegistry()
    registry.assign([voice(0)], [1.2])
    for _ in range(4):
        registry.assign([voice(0)], [.3], reference_eligible=False, embedding_durations=quality)
    assert registry.speakers[0].learned_observations == 0
    assert len(FeatureBank.reference_samples(registry.feature_bank.samples, registry.speakers)) == 1


@pytest.mark.parametrize('quality', [[float('nan')], [float('inf')], [.1], [], [-1]])
def test_invalid_context_duration_is_rejected_before_mutation(quality):
    registry = SpeakerRegistry()
    with pytest.raises(ValueError):
        registry.assign([voice(0)], [.3], embedding_durations=quality)
    assert not registry.speakers and not registry.feature_bank.samples


def test_older_database_without_context_column_uses_current_seconds(tmp_path):
    path = tmp_path / 'people.sqlite3'
    store = ProfileStore(path)
    person = store.create(voice(0), 1, .3)
    store.add_samples([(voice(0), .3, person.id)])
    with store.connection() as db:
        db.execute('ALTER TABLE voice_samples DROP COLUMN embedding_seconds')
    restored = ProfileStore(path)
    sample = restored.load_samples()[0]
    assert sample.evidence_seconds == .3 and sample.embedding_seconds is None
    assert not FeatureBank.reference_samples([sample], restored.load_speakers())


def test_separator_permutation_does_not_choose_a_different_enrollment_anchor():
    first, second = SpeakerRegistry(), SpeakerRegistry()
    a = voice(1)
    b = .4 * voice(1) + np.sqrt(.84) * voice(2)
    for registry in (first, second):
        registry.assign([voice(0)], [1.2])
    for _ in range(2):
        forward = first.assign([a, b], [1.2, 1.0], embedding_durations=[2.4, 2.1])
        reverse = second.assign([b, a], [1.0, 1.2], embedding_durations=[2.1, 2.4])
        assert forward == reverse[::-1]
        assert first.decisions == second.decisions[::-1]
    assert forward == ['speaker_2', 'speaker_2']
    np.testing.assert_array_equal(first.speakers[1].vector, second.speakers[1].vector)
    np.testing.assert_array_equal(first.speakers[1].vector, a)
