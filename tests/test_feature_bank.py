import numpy as np

from selective_hearing.core import Speaker, SpeakerRegistry
from selective_hearing.feature_bank import FeatureBank, MAX_PER_SPEAKER, MAX_PENDING, MAX_SAMPLES
from selective_hearing.profiles import ProfileStore


def voice(degrees):
    v = np.zeros(192, np.float32)
    v[:2] = [np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))]
    return v


def test_neighborhood_recovers_variant_without_one_outlier_taking_over():
    speaker = Speaker('speaker_1', voice(0))
    bank = FeatureBank()
    bank.add(voice(90), 1, speaker.id)
    assert bank.scores(voice(90), [speaker])[0] < .1
    bank = FeatureBank()
    for angle in (40, 45, 50):
        bank.add(voice(angle), 1, speaker.id)
    assert speaker.similarity(voice(70)) < .45
    assert bank.scores(voice(70), [speaker])[0] > .85


def test_repeated_weak_predictions_cannot_absorb_a_different_person(tmp_path):
    store = ProfileStore(tmp_path / 'profiles.sqlite3')
    registry = SpeakerRegistry(store=store)
    registry.assign([voice(0)], [1.2])
    # Old short-frame matching accepts these borderline observations. They
    # must not become trusted votes for an unrelated 90-degree voice.
    for _ in range(4):
        registry.assign([voice(60)], [1.2])
    again = SpeakerRegistry(store=ProfileStore(store.path))
    assert again.assign([voice(90)], [1.2]) == ['speaker_2']
    assert again.assign([voice(90)], [1.2]) == ['speaker_2']
    assert len(again.feature_bank.reference_samples(again.feature_bank.samples, again.speakers)) < len(again.feature_bank.samples)


def test_weak_samples_do_not_relabel_history_or_short_samples_vote():
    speakers = [Speaker('speaker_1', voice(0))]
    bank = FeatureBank()
    for _ in range(3):
        bank.add(voice(60), 1.2, 'speaker_1')
    pending = bank.add(voice(90), 1.2, None)
    assert not bank.reconcile(speakers) and pending.speaker_id is None
    short = FeatureBank()
    for _ in range(3):
        short.add(voice(40), .5, 'speaker_1')
    assert short.scores(voice(70), speakers)[0] < .45


def test_old_learned_centroid_does_not_override_a_weak_anchor_alone():
    speaker = Speaker('speaker_1', voice(0), learned_vector=voice(45))
    assert speaker.similarity(voice(90)) < .45


def test_history_can_move_to_new_cluster_with_names_and_ids_preserved(tmp_path):
    store = ProfileStore(tmp_path / 'profiles.sqlite3')
    a, b = store.create(voice(0), 1, 1), store.create(voice(90), 1, 1)
    store.rename(a.id, 'A'); store.rename(b.id, 'B')
    store.add_samples([(voice(70), 1, a.id)])  # an earlier misclassification
    store.add_samples([(voice(72), 1, None)])  # an earlier pending sample
    for angle in (80, 85, 90):
        store.add_samples([(voice(angle), 1, b.id)])
    bank = FeatureBank(store.load_samples())
    changes = bank.reconcile(store.load_speakers())
    assert len(changes) == 2
    assert changes[0][1:] == (a.id, b.id)
    store.correct_samples(changes)
    restored = ProfileStore(store.path)
    assert all(sample.speaker_id == b.id for sample in restored.load_samples())
    assert [(s['id'], s['name']) for s in restored.summaries()] == [(a.id, 'A'), (b.id, 'B')]
    assert restored.summaries()[1]['feature_samples'] == 5


def test_correction_does_not_self_vote_or_collapse_simultaneous_sources():
    speakers = [Speaker('speaker_1', voice(0)), Speaker('speaker_2', voice(90))]
    bank = FeatureBank()
    bad = bank.add(voice(70), 1, 'speaker_1', batch_id=1)
    bank.add(voice(90), 1, 'speaker_2', batch_id=1)
    for _ in range(3):
        bank.add(voice(85), 1, 'speaker_2')
    bank.reconcile(speakers)
    assert bad.speaker_id == 'speaker_1'
    # A sample cannot provide its own third supporting neighbor.
    small = FeatureBank()
    for _ in range(2):
        small.add(voice(85), 1, 'speaker_2')
    pending = small.add(voice(80), 1, None)
    assert not small.reconcile(speakers) and pending.speaker_id is None


def test_feature_memory_is_bounded_and_deleted_person_loses_samples(tmp_path):
    store = ProfileStore(tmp_path / 'profiles.sqlite3')
    person = store.create(voice(0), 1, 1)
    store.add_samples([(voice(0), 1, person.id)] * (MAX_PER_SPEAKER + 20))
    store.add_samples([(voice(90), 1, None)] * (MAX_PENDING + 20))
    store.correct_samples([])
    bank = FeatureBank(store.load_samples())
    assert len(bank.samples) == MAX_PER_SPEAKER + MAX_PENDING
    store.delete(person.id)
    assert all(s.speaker_id is None for s in store.load_samples())
    for i in range(MAX_SAMPLES * 2):
        bank.add(voice(0), 1, f'speaker_{i}')
    bank.trim()
    assert len(bank.samples) == MAX_SAMPLES


def test_concurrent_append_ids_and_stale_correction_do_not_overwrite(tmp_path):
    path = tmp_path / 'profiles.sqlite3'
    first, second = ProfileStore(path), ProfileStore(path)
    people = [first.create(voice(angle), 1, 1) for angle in (0, 45, 90)]
    first.add_samples([(voice(0), 1, people[0].id)])
    second.add_samples([(voice(90), 1, people[2].id)])
    samples = first.load_samples()
    assert samples[0].id != samples[1].id
    first.correct_samples([(samples[0].id, people[0].id, people[1].id)])
    second.correct_samples([(samples[0].id, people[0].id, people[2].id)])
    assert first.load_samples()[0].speaker_id == people[1].id


def test_pending_features_survive_restart_and_old_profile_migrates(tmp_path):
    path = tmp_path / 'profiles.sqlite3'
    store = ProfileStore(path)
    person = store.create(voice(0), 1, 1)
    with store.connection() as db:
        db.execute('DROP TABLE voice_samples')  # simulate the previous schema
    restored = ProfileStore(path)
    assert restored.load_speakers()[0].id == person.id
    registry = SpeakerRegistry(store=restored)
    assert registry.assign([voice(90)], [.8]) == [None]
    again = SpeakerRegistry(store=ProfileStore(path))
    assert len(again.feature_bank.samples) == 1
    assert again.feature_bank.samples[0].speaker_id is None
    assert len(again.candidates) == 1
    assert again.assign([voice(90)], [1.2]) == ['speaker_2']
