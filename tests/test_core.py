import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry, AudioPacket, PlaybackBuffer, OutputControl, resample, unit


def test_output_permutation_does_not_swap_identity():
    registry = SpeakerRegistry()
    a, b = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    assert registry.assign([a, b], [1., 1.]) == ["speaker_1", None]
    assert not any(s["ready"] for s in registry.snapshot())
    assert registry.assign([b, a], [1., 1.]) == ["speaker_2", "speaker_1"]
    assert all(s["ready"] for s in registry.snapshot())


def test_unknown_voice_never_forced_to_selected_identity():
    registry = SpeakerRegistry()
    registry.assign([[1, 0, 0], [0, 1, 0]], [2, 2])
    registry.assign([[1, 0, 0], [0, 1, 0]], [2, 2])
    assert registry.assign([[0, 0, 1]], [2]) == [None]
    assert registry.assign([[0, 0, 1]], [2]) == ['speaker_3']
    assert len(registry.speakers) == 3
    assert registry.assign([[0, 0, 1], [1, 0, 0]], [2, 2]) == ['speaker_3', 'speaker_1']


def test_short_unmatched_fragment_does_not_create_extra_person():
    registry = SpeakerRegistry()
    registry.assign([[1, 0, 0], [0, 1, 0]], [2, 2])
    registry.assign([[1, 0, 0], [0, 1, 0]], [2, 2])
    assert registry.assign([[0, 0, 1]], [.8]) == [None]
    assert registry.assign([[1, 0, 0]], [.8]) == ['speaker_1']  # can still match an existing identity
    assert registry.assign([[0, 0, 1]], [1.2]) == [None]
    assert registry.assign([[0, 0, 1]], [1.2]) == ['speaker_3']


def test_isolated_outliers_never_accumulate_into_people():
    registry = SpeakerRegistry()
    vectors = np.eye(40)
    for _ in range(2):
        registry.assign(vectors[:2], [1.2, 1.2])
    for vector in vectors[2:]:
        assert registry.assign([vector], [1.2]) == [None]
        assert registry.assign(vectors[:2], [1.2, 1.2]) == ['speaker_1', 'speaker_2']
    assert len(registry.speakers) == 2
    assert len(registry.candidates) <= registry.max_candidates


def test_candidate_requires_distinct_blocks_and_expires():
    registry = SpeakerRegistry()
    registry.assign([[1, 0]], [1.2])
    assert registry.assign([[0, 1], [0, 1]], [1.2, 1.2]) == [None, None]
    assert len(registry.speakers) == len(registry.candidates) == 1
    for _ in range(registry.candidate_blocks + 1):
        registry.assign([], [])
    assert not registry.candidates
    assert registry.assign([[0, 1]], [1.2]) == [None]
    assert registry.assign([[0, 1]], [1.2]) == ['speaker_2']


def test_short_voice_drift_matches_without_lowering_new_person_evidence():
    registry = SpeakerRegistry()
    registry.assign([[1, 0, 0]], [1.2])
    # 0.48 cosine: short-frame recognition can recover the existing person.
    changed = [0.48, np.sqrt(1 - .48 ** 2), 0]
    assert registry.assign([changed], [1.2]) == ['speaker_1']
    # The uncertainty band neither creates a person nor routes to a known one.
    uncertain = [0.35, 0, np.sqrt(1 - .35 ** 2)]
    for _ in range(12):
        assert registry.assign([uncertain], [1.2]) == [None]
    assert len(registry.speakers) == 1 and not registry.candidates


def test_pending_person_is_competitor_for_ambiguous_known_match():
    registry = SpeakerRegistry()
    registry.assign([[1, 0], [0, 1]], [1.2, 1.2])
    assert registry.assign([[1, 1]], [1.2]) == [None]


def test_averaged_candidate_is_rechecked_against_existing_person():
    registry = SpeakerRegistry()
    registry.assign([[1, 0, 0]], [1.2])
    radius = np.sqrt(1 - .29 ** 2)
    assert registry.assign([[.29, radius, 0]], [1.2]) == [None]
    # Both individual fragments are below the novelty threshold, but their
    # aggregate is not. This must not create another identity for a weak voice.
    assert registry.assign([[.29, radius * .6, radius * .8]], [1.2]) == [None]
    assert len(registry.speakers) == 1 and not registry.candidates


def test_independent_tracks_keep_other_choices_and_missing_tracks_silent():
    control = OutputControl()
    packet = AudioPacket(np.full(5, .9, np.float32), {
        'speaker_1': np.full(5, .1, np.float32),
        'speaker_2': np.full(5, .2, np.float32),
        'speaker_3': np.full(5, .3, np.float32)})
    control.set_track('speaker_1', False)
    np.testing.assert_allclose(packet.select(control.get()[0]), .5)
    snapshot = control.get()[0]
    control.set_track('speaker_2', False)
    control.set_track('speaker_1', True)
    np.testing.assert_allclose(packet.select(control.get()[0]), .4)  # listening to A keeps B muted
    np.testing.assert_allclose(packet.select(snapshot), .5)  # callback snapshot is immutable
    control.set(mode='mute')
    control.set_track('speaker_1', True)
    assert not control.track_enabled('speaker_4')
    np.testing.assert_allclose(packet.select(control.get()[0]), .1)
    packet.tracks.pop('speaker_1')
    np.testing.assert_array_equal(packet.select(control.get()[0]), 0)  # no original/background fallback
    control.set(mode='all')
    assert control.track_enabled('speaker_4')
    np.testing.assert_allclose(packet.select(control.get()[0]), .9)


@pytest.mark.parametrize('has_track', [False, True])
def test_undo_last_exclusion_restores_original_even_when_identity_is_missing(has_track):
    control = OutputControl()
    original = np.full(320, .3, np.float32)
    packet = AudioPacket(original, {'speaker_1': original * .5} if has_track else {})
    # Clicking an already enabled person must not change the audio either.
    control.set_track('speaker_1', True)
    np.testing.assert_array_equal(packet.select(control.get()[0]), original)
    for _ in range(3):
        control.set_track('speaker_1', False)
        np.testing.assert_array_equal(packet.select(control.get()[0]), 0)
        control.set_track('speaker_1', True)
        np.testing.assert_array_equal(packet.select(control.get()[0]), original)


def test_unmuting_one_person_keeps_remaining_exclusions_and_global_mute_policy():
    control = OutputControl()
    packet = AudioPacket(np.full(320, .3, np.float32))  # unknown voice
    control.set_track('speaker_1', False)
    control.set_track('speaker_2', False)
    control.set_track('speaker_1', True)
    assert not control.track_enabled('speaker_2')
    np.testing.assert_array_equal(packet.select(control.get()[0]), 0)
    control.set_track('speaker_2', True)
    np.testing.assert_array_equal(packet.select(control.get()[0]), packet.original)
    control.set(mode='mute')
    control.set_track('speaker_1', True)
    assert not control.track_enabled('speaker_2')
    np.testing.assert_array_equal(packet.select(control.get()[0]), 0)
    packet.tracks['speaker_1'] = np.full(320, .1, np.float32)
    np.testing.assert_allclose(packet.select(control.get()[0]), .1)
    control.set_track('speaker_1', False)
    np.testing.assert_array_equal(packet.select(control.get()[0]), 0)


def test_two_simultaneous_tracks_cannot_share_identity():
    registry = SpeakerRegistry()
    registry.assign([[1, 0]], [2])
    assigned = registry.assign([[1, 0], [0.99, 0.01]], [2, 2])
    assert assigned.count("speaker_1") == 1
    assert len(registry.speakers) == 1


def test_ambiguous_match_does_not_create_duplicate():
    registry = SpeakerRegistry()
    registry.assign([[1, 0], [0, 1]], [2, 2])
    assert registry.assign([[1, 1]], [2]) == [None]


def test_mute_consumes_timeline_and_switch_is_immediate():
    buffer = PlaybackBuffer(100)
    buffer.put(AudioPacket(np.arange(10, dtype=np.float32) / 20,
                           {"speaker_1": -np.arange(10, dtype=np.float32) / 20}))
    np.testing.assert_allclose(buffer.read(3, "mute", 1, 0), 0)
    np.testing.assert_allclose(buffer.read(2, "speaker_1", 1, 0), [-.15, -.2])
    np.testing.assert_allclose(buffer.read(2, "all", 1, 0), [.25, .3])


def test_selected_target_absence_is_silence_not_mixture():
    packet = AudioPacket(np.ones(20, dtype=np.float32))
    assert np.all(packet.select("speaker_1") == 0)


def test_excluding_first_person_is_silent_until_another_identified_person_arrives():
    packet = AudioPacket(np.full(10, .9, np.float32), {'speaker_1': np.full(10, .2, np.float32)})
    np.testing.assert_array_equal(packet.select('exclude:speaker_1'), 0)
    packet.tracks['speaker_2'] = np.arange(10, dtype=np.float32) / 20
    np.testing.assert_allclose(packet.slice('exclude:speaker_1', 2, 5), [.1, .15, .2])
    packet.tracks.clear()
    np.testing.assert_array_equal(packet.select('exclude:speaker_1'), 0)  # no mixture leakage


def test_bounded_buffer_drops_oldest_and_survives_partial_read():
    buffer = PlaybackBuffer(10)
    buffer.put(AudioPacket(np.ones(8, np.float32) * .1))
    buffer.read(3, "all", 1, 0)
    buffer.put(AudioPacket(np.ones(8, np.float32) * .2))
    assert buffer.dropped == 1
    assert buffer.samples == 8
    np.testing.assert_allclose(buffer.read(10, "all", 1, 0), [.2] * 8 + [0] * 2)
    assert buffer.samples == 0


def test_silence_and_invalid_embeddings():
    registry = SpeakerRegistry()
    assert registry.assign([], []) == []
    with pytest.raises(ValueError):
        unit([0, 0])
    with pytest.raises(ValueError):
        unit([np.nan, 1])


def test_resampling_preserves_duration():
    wave = np.sin(np.arange(48000) * 2 * np.pi * 440 / 48000).astype(np.float32)
    output = resample(wave, 48000, 16000)
    assert len(output) == 16000
    assert output.dtype == np.float32
