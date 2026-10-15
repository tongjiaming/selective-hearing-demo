import sqlite3
import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry
from selective_hearing.profiles import ProfileStore


def voice(angle=0, axis=1):
    vector = np.zeros(192, np.float32)
    vector[0], vector[axis] = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return vector


def test_update_resets_learning_but_restart_and_window_settings_survive(tmp_path):
    path = tmp_path / 'people.sqlite3'
    old = ProfileStore(path, revision='v1')
    person = old.create(voice(), 1, 1.2)
    old.add_samples([(voice(), 1.2, person.id), (voice(90), .5, None)])
    old.set_window_seconds(1.5)
    assert len(ProfileStore(path, revision='v1').load_samples()) == 2
    updated = ProfileStore(path, revision='v2')
    assert updated.learning_reset
    assert not updated.load_speakers() and not updated.load_samples()
    assert updated.window_seconds(3) == 1.5
    with pytest.raises(ValueError, match='旧窗口不能再写入'):
        old.create(voice(), 1, 1.2)
    new = updated.create(voice(), 1, 1.2)
    assert new.id != person.id and updated.summaries()[0]['name'] == '说话人 1'
    reopened = ProfileStore(path, revision='v2')
    assert not reopened.learning_reset and len(reopened.load_speakers()) == 1


def test_legacy_learning_is_cleared_on_first_updated_launch(tmp_path):
    path = tmp_path / 'people.sqlite3'
    old = ProfileStore(path)
    old.create(voice(), 1, 1.2)
    with old.connection() as db:
        db.execute("DELETE FROM metadata WHERE key='learning_revision'")
    updated = ProfileStore(path)
    assert updated.learning_reset and not updated.load_speakers()


def test_reopen_restores_identity_name_and_provisional_state(tmp_path):
    path = tmp_path / 'people.sqlite3'
    store = ProfileStore(path)
    registry = SpeakerRegistry(store=store)
    assert registry.assign([voice()], [1.2]) == ['speaker_1']
    store.rename('speaker_1', '小王')
    reopened = ProfileStore(path)
    assert reopened.summaries()[0]['name'] == '小王'
    assert not reopened.summaries()[0]['ready']
    again = SpeakerRegistry(store=reopened)
    assert again.assign([voice()], [1.2]) == ['speaker_1']
    assert reopened.summaries()[0]['ready']
    assert len(reopened.load_speakers()) == 1


def test_provisional_profiles_survive_without_creating_extra_profile_on_load(tmp_path):
    store = ProfileStore(tmp_path / 'people.sqlite3')
    registry = SpeakerRegistry(store=store)
    registry.assign([voice()], [1.2])
    assert registry.assign([voice(90)], [1.2]) == [None]
    assert registry.assign([voice(90)], [1.2]) == ['speaker_2']
    assert not registry.candidates
    restored = SpeakerRegistry(store=store)
    assert not restored.candidates and len(restored.speakers) == 2
    assert restored.assign([voice(90)], [1.2]) == ['speaker_2']
    assert restored.assign([voice(90)], [1.2]) == ['speaker_2']


def test_learning_improves_unseen_variant_and_survives_restart(tmp_path):
    path = tmp_path / 'people.sqlite3'
    adaptive = SpeakerRegistry(store=ProfileStore(path))
    frozen = SpeakerRegistry(learning=False)
    for registry in (adaptive, frozen):
        registry.assign([voice()], [1.2])
        for _ in range(4):
            assert registry.assign([voice(45)], [1.2]) == ['speaker_1']
    assert frozen.assign([voice(70)], [1.2]) == [None]
    restored = SpeakerRegistry(store=ProfileStore(path))
    assert restored.assign([voice(70)], [1.2]) == ['speaker_1']
    assert restored.speakers[0].learned_observations == 4  # anchor guard blocks risky learning
    np.testing.assert_array_equal(restored.speakers[0].vector, voice())


def test_borderline_match_does_not_train_profile(tmp_path):
    store = ProfileStore(tmp_path / 'people.sqlite3')
    registry = SpeakerRegistry(store=store)
    registry.assign([voice()], [1.2])
    assert registry.assign([voice(60)], [1.2]) == ['speaker_1']
    saved = store.load_speakers()[0]
    assert saved.learned_vector is None and saved.learned_observations == 0


def test_delete_does_not_reuse_id_or_resurrect_stale_worker(tmp_path):
    path = tmp_path / 'people.sqlite3'
    store = ProfileStore(path)
    registry = SpeakerRegistry(store=store)
    registry.assign([voice()], [1.2])
    store.delete('speaker_1')
    with pytest.raises(ValueError, match='已被删除'):
        registry.assign([voice()], [1.2])
    assert not store.load_speakers()
    restarted = SpeakerRegistry(store=ProfileStore(path))
    assert restarted.assign([voice()], [1.2]) == ['speaker_2']
    assert restarted.snapshot()[0]['name'] == '说话人 1'
    assert restarted.snapshot()[0]['display_number'] == 1


def test_display_numbers_fill_holes_without_renumbering_existing_people(tmp_path):
    store = ProfileStore(tmp_path / 'people.sqlite3')
    people = [store.create(voice(), 1, 1) for _ in range(3)]
    store.rename(people[2].id, '小王')
    store.delete(people[1].id)
    new = store.create(voice(), 1, 1)
    assert new.id == 'speaker_4'
    rows = store.summaries()
    assert [(r['display_number'], r['name']) for r in rows] == [(1, '说话人 1'), (3, '小王'), (2, '说话人 2')]
    assert store.rename(new.id, ' ') == '说话人 2'
    assert ProfileStore(store.path).summaries() == rows


def test_old_default_speaker_six_becomes_one_preserving_identity_and_custom_names(tmp_path):
    store = ProfileStore(tmp_path / 'people.sqlite3')
    people = [store.create(voice(), 1, 1) for _ in range(7)]
    for person in people[:5]:
        store.delete(person.id)
    store.rename(people[6].id, '小王')
    with store.connection() as db:
        db.execute('DROP INDEX speaker_display_number')
        db.execute('ALTER TABLE speakers DROP COLUMN display_number')
    restored = ProfileStore(store.path)
    assert [(r['id'], r['display_number'], r['name']) for r in restored.summaries()] == [
        ('speaker_6', 1, '说话人 1'), ('speaker_7', 2, '小王')]
    assert ProfileStore(store.path).summaries() == restored.summaries()


def test_worker_learning_does_not_overwrite_rename_and_stats(tmp_path):
    path = tmp_path / 'people.sqlite3'
    store = ProfileStore(path)
    first = SpeakerRegistry(store=store)
    first.assign([voice()], [1.2])
    second = SpeakerRegistry(store=ProfileStore(path))
    ProfileStore(path).rename('speaker_1', 'Alice')
    first.assign([voice(40)], [1.2])
    second.assign([voice(40)], [1.2])
    saved = store.summaries()[0]
    assert saved['name'] == 'Alice'
    assert saved['observations'] == 3 and saved['learned_observations'] == 2


@pytest.mark.parametrize('problem', ['model', 'vector', 'corrupt'])
def test_bad_profile_never_silently_resets_or_overwrites(tmp_path, problem):
    path = tmp_path / 'people.sqlite3'
    store = ProfileStore(path)
    store.create(voice(), 1, 1.2)
    if problem == 'corrupt':
        path.write_bytes(b'not a sqlite database')
    else:
        with sqlite3.connect(path) as db:
            if problem == 'model':
                db.execute("UPDATE metadata SET value='other-model' WHERE key='model'")
            else:
                db.execute('UPDATE speakers SET vector=?', (np.ones(3, np.float32).tobytes(),))
    before = path.read_bytes()
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        ProfileStore(path)
    assert path.read_bytes() == before


def test_file_library_is_separate_from_live_library(tmp_path):
    live = ProfileStore(tmp_path / 'live.sqlite3')
    offline = ProfileStore(tmp_path / 'file.sqlite3')
    live.create(voice(), 1, 1.2)
    assert offline.summaries() == []


def test_analysis_window_persists_and_rejects_invalid_values(tmp_path):
    path = tmp_path / 'settings.sqlite3'
    store = ProfileStore(path)
    assert store.window_seconds(3) == 3
    store.set_window_seconds('2.5')
    assert ProfileStore(path).window_seconds(3) == 2.5
    for invalid in ['', 'abc', 'nan', 'inf', '0', '-1', '11']:
        with pytest.raises(ValueError, match='1 到 10'):
            store.set_window_seconds(invalid)
        assert store.window_seconds(3) == 2.5
