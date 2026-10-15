import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry, OutputControl, AudioPacket
from selective_hearing.profiles import ProfileStore


def voice(angle):
    vector = np.zeros(192, np.float32)
    vector[:2] = [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
    return vector


def repair_fixture(tmp_path, *, references=3, conflict=False):
    store = ProfileStore(tmp_path / 'people.sqlite3')
    draft = store.create(voice(30), 1, .25)
    target = store.create(voice(0), 3, 3)
    store.add_samples([(voice(30), .25, draft.id)])
    for _ in range(references):
        store.add_samples([(voice(0), 1, target.id)])
    if conflict:
        store.record_conflict(draft.id, target.id)
    return store, draft, target


def test_first_and_later_short_voices_are_immediately_saved_and_selectable(tmp_path):
    store = ProfileStore(tmp_path / 'people.sqlite3')
    registry = SpeakerRegistry(store=store)
    assert registry.assign([voice(0), voice(90)], [.2, .25]) == ['speaker_1', 'speaker_2']
    assert len(store.load_speakers()) == 2 and not any(s.ready for s in registry.speakers)
    assert registry.assign([voice(180)], [.2]) == ['speaker_3']
    assert registry.assign([voice(90)], [.3]) == ['speaker_2']
    assert len(SpeakerRegistry(store=ProfileStore(store.path)).speakers) == 3


def test_stronger_later_evidence_repairs_draft_and_persists_alias_and_name(tmp_path):
    store, draft, target = repair_fixture(tmp_path)
    store.rename(draft.id, '小王')
    registry = SpeakerRegistry(store=store)
    registry.assign([], [])
    assert registry.last_merges == [{'from': draft.id, 'to': target.id}]
    assert [s.id for s in registry.speakers] == [target.id]
    assert all(s.speaker_id == target.id for s in store.load_samples())
    assert store.summaries()[0]['name'] == '小王'
    restored = SpeakerRegistry(store=ProfileStore(store.path))
    assert restored.aliases[draft.id] == target.id
    assert restored.assign([voice(0)], [.3]) == [target.id]


def test_repeated_short_matches_do_not_make_a_wrong_draft_irreparable(tmp_path):
    store, draft, target = repair_fixture(tmp_path)
    for _ in range(5):
        store.record_match(draft, voice(30), .25, False)
        store.add_samples([(voice(30), .25, draft.id)])
    assert draft.ready
    registry = SpeakerRegistry(store=store)
    registry.assign([], [])
    assert registry.aliases[draft.id] == target.id
    assert len(store.load_speakers()) == 1


def test_close_existing_choices_do_not_create_another_profile_or_train_on_ambiguity():
    registry = SpeakerRegistry()
    registry.assign([voice(0), voice(60)], [1, 1])
    assigned = registry.assign([voice(29)], [.3])
    assert assigned == ['speaker_1'] and len(registry.speakers) == 2
    assert not any(s.learned_observations for s in registry.speakers)


@pytest.mark.parametrize('references,conflict', [(2, False), (3, True)])
def test_insufficient_evidence_or_simultaneous_people_do_not_merge(tmp_path, references, conflict):
    store, draft, target = repair_fixture(tmp_path, references=references, conflict=conflict)
    registry = SpeakerRegistry(store=store)
    registry.assign([], [])
    assert not registry.last_merges and len(store.load_speakers()) == 2
    assert store.load_samples()[0].speaker_id == draft.id


def test_different_custom_names_are_not_automatically_consolidated(tmp_path):
    store, draft, target = repair_fixture(tmp_path)
    store.rename(draft.id, '甲')
    store.rename(target.id, '乙')
    registry = SpeakerRegistry(store=store)
    registry.assign([], [])
    assert not registry.last_merges and len(store.load_speakers()) == 2
    assert {s['name'] for s in store.summaries()} == {'甲', '乙'}


def test_merge_preserves_mute_for_queued_old_ids_and_can_be_undone_by_listen():
    control = OutputControl()
    packet = AudioPacket(np.full(10, .3, np.float32), {
        'speaker_1': np.full(10, .1, np.float32), 'speaker_2': np.full(10, .2, np.float32)})
    control.set_track('speaker_1', False)
    old_snapshot = control.get()[0]
    control.apply_aliases({'speaker_1': 'speaker_2'})
    np.testing.assert_allclose(packet.select(control.get()[0]), 0, atol=1e-7)
    np.testing.assert_allclose(packet.select(old_snapshot), .2)
    assert not control.track_enabled('speaker_2')
    control.set_track('speaker_2', True)
    control.apply_aliases({'speaker_1': 'speaker_2'})  # repeated progress must not re-mute
    np.testing.assert_allclose(packet.select(control.get()[0]), .3)
    control.apply_aliases({'speaker_2': 'speaker_3'})
    control.set_track('speaker_3', False)
    np.testing.assert_allclose(packet.select(control.get()[0]), 0, atol=1e-7)
    assert control.canonical('speaker_1') == 'speaker_3'


def test_deleting_merged_profile_removes_aliases_and_features(tmp_path):
    store, draft, target = repair_fixture(tmp_path)
    SpeakerRegistry(store=store).assign([], [])
    store.delete(target.id)
    assert not store.load_aliases() and not store.load_samples() and not store.load_speakers()


def test_file_history_moves_to_surviving_track_when_draft_is_repaired(tmp_path):
    import soundfile as sf
    from selective_hearing.audio import process_file
    store, draft, target = repair_fixture(tmp_path, references=2)
    class Backend:
        block = 0
        def separate(self, audio):
            self.block += 1
            return [audio]
        def speech(self, audio): return audio[:4000] if self.block == 1 else audio
        def embedding(self, audio): return voice(30 if self.block == 1 else 0)
    path = tmp_path / 'input.wav'
    sf.write(path, np.full(32000, .1, np.float32), 16000, subtype='FLOAT')
    events = []
    packet, report = process_file(path, Backend(), notify=events.append,
                                  window_seconds=1, registry=SpeakerRegistry(store=store))
    assert events[0]['active'] == [draft.id]
    assert events[1]['profile_merges'] == [{'from': draft.id, 'to': target.id}]
    assert list(packet.tracks) == [target.id]
    np.testing.assert_allclose(packet.tracks[target.id], .1)
    assert [s['id'] for s in report['speakers']] == [target.id]
