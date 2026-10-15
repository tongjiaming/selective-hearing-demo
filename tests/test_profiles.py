import sqlite3
import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry
from selective_hearing.profiles import ProfileStore


def voice(angle=0, axis=1):
    vector = np.zeros(192, np.float32)
    vector[0], vector[axis] = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return vector


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


def test_pending_features_survive_without_creating_profile_on_load(tmp_path):
    store = ProfileStore(tmp_path / 'people.sqlite3')
    registry = SpeakerRegistry(store=store)
    registry.assign([voice()], [1.2])
    assert registry.assign([voice(90)], [1.2]) == [None]
    assert len(registry.candidates) == 1
    restored = SpeakerRegistry(store=store)
    assert len(restored.candidates) == 1 and len(restored.speakers) == 1
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
