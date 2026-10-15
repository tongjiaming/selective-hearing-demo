import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry, AudioPacket, PlaybackBuffer, OutputControl, SpeakingActivity, resample, unit, UNCLASSIFIED_ID


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
    assert registry.assign([[0, 0, 1]], [1.2]) == ['speaker_3']  # .8 + 1.2 seconds now accumulate
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
    # The former dead zone now accumulates conservative, repeated evidence.
    uncertain = [0.35, 0, np.sqrt(1 - .35 ** 2)]
    assert registry.assign([uncertain], [1.2]) == [None]
    assert registry.assign([uncertain], [1.2]) == [None]
    assert registry.assign([uncertain], [1.2]) == ['speaker_2']
    assert len(registry.speakers) == 2 and not registry.candidates


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
    assert len(registry.speakers) == 1 and registry.candidates[0].borderline
    assert registry.candidates[0].consistent_observations == 1
    # A return to the original voice must still win and cannot create a third person.
    assert registry.assign([[1, 0, 0]], [1.2]) == ['speaker_1']


def test_promoted_voice_retires_old_candidates_so_they_cannot_mute_it():
    # Two initially inconsistent observations of B create separate candidates.
    # Once B has a confirmed profile, the older variant belongs to its
    # uncertainty band and must no longer act as a competing new person.
    registry = SpeakerRegistry()
    a = np.array([1., 0., 0., 0.])
    b = np.array([0., 1., 0., 0.])
    variant = unit([0., .48, np.sqrt(1 - .48 ** 2), 0.])
    registry.assign([a, b], [1.2, 1.2])
    registry.assign([variant], [1.2])
    assert len(registry.candidates) == 2
    assert registry.assign([b], [1.2]) == ['speaker_2']
    assert not registry.candidates
    assert registry.assign([variant], [1.2]) == ['speaker_2']
    assert len(registry.speakers) == 2
    # A truly distinct C must still be confirmed, not merged into B.
    c = [0., 0., 0., 1.]
    assert registry.assign([c], [1.2]) == [None]
    assert registry.assign([c], [1.2]) == ['speaker_3']


def test_independent_tracks_keep_other_choices_and_missing_tracks_silent():
    control = OutputControl()
    packet = AudioPacket(np.full(5, .9, np.float32), {
        'speaker_1': np.full(5, .1, np.float32),
        'speaker_2': np.full(5, .2, np.float32),
        'speaker_3': np.full(5, .3, np.float32)})
    control.set_track('speaker_1', False)
    np.testing.assert_allclose(packet.select(control.get()[0]), .8)
    snapshot = control.get()[0]
    control.set_track('speaker_2', False)
    control.set_track('speaker_1', True)
    np.testing.assert_allclose(packet.select(control.get()[0]), .7)  # A and the remainder stay on; B stays muted
    np.testing.assert_allclose(packet.select(snapshot), .8)  # callback snapshot is immutable
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
        np.testing.assert_allclose(packet.select(control.get()[0]), original * (.5 if has_track else 1))
        control.set_track('speaker_1', True)
        np.testing.assert_array_equal(packet.select(control.get()[0]), original)


def test_unmuting_one_person_keeps_remaining_exclusions_and_global_mute_policy():
    control = OutputControl()
    packet = AudioPacket(np.full(320, .3, np.float32))  # unknown voice
    control.set_track('speaker_1', False)
    control.set_track('speaker_2', False)
    control.set_track('speaker_1', True)
    assert not control.track_enabled('speaker_2')
    np.testing.assert_array_equal(packet.select(control.get()[0]), packet.original)
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


@pytest.mark.parametrize('mask', range(16))
def test_people_and_unclassified_are_four_independent_tracks(mask):
    rng = np.random.default_rng(45)
    a, b, c, unclassified = rng.normal(0, .03, (4, 32)).astype(np.float32)
    parts = [a, b, c, unclassified]
    ids = ['speaker_1', 'speaker_2', 'speaker_3', UNCLASSIFIED_ID]
    original = sum(parts)
    packet = AudioPacket(original, dict(zip(ids[:3], parts[:3])))
    original_copy = original.copy()
    control = OutputControl()
    for index, identity in enumerate(ids):
        control.set_track(identity, bool(mask & (1 << index)))
    expected = sum((track for index, track in enumerate(parts) if mask & (1 << index)), np.zeros(32, np.float32))
    np.testing.assert_allclose(packet.slice(control.get()[0], 5, 17), expected[5:17], atol=2e-8)
    np.testing.assert_array_equal(packet.original, original_copy)
    assert packet.active(5, 17) == set(ids[:3])  # no fake speaker/activity for background


def test_all_people_off_keeps_unknown_until_its_own_switch_or_global_mute():
    control = OutputControl()
    packet = AudioPacket(np.full(10, .5, np.float32), {'speaker_1': np.full(10, .2, np.float32)})
    control.set_track('speaker_1', False)
    np.testing.assert_allclose(packet.select(control.get()[0]), .3)
    control.set_track(UNCLASSIFIED_ID, False)
    np.testing.assert_array_equal(packet.select(control.get()[0]), 0)
    control.set_track(UNCLASSIFIED_ID, True)
    np.testing.assert_allclose(packet.select(control.get()[0]), .3)
    assert not control.track_enabled('speaker_1')
    control.set(mode='mute')
    assert not control.track_enabled(UNCLASSIFIED_ID)
    np.testing.assert_array_equal(packet.select(control.get()[0]), 0)
    control.set_track(UNCLASSIFIED_ID, True)
    assert not control.track_enabled('speaker_1') and not control.track_enabled('speaker_2')
    np.testing.assert_allclose(packet.select(control.get()[0]), .3)


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


def test_speaking_activity_tracks_timeline_even_when_muted_and_expires():
    buffer = PlaybackBuffer(1280)
    a = np.concatenate([np.full(320, .1, np.float32), np.zeros(320, np.float32)])
    b = a[::-1].copy()
    buffer.put(AudioPacket(a + b, {'speaker_1': a, 'speaker_2': b}))
    # Merely enqueueing an analyzed block must not light up the speaker yet.
    assert not buffer.activity.active(1)
    np.testing.assert_array_equal(buffer.read(320, 'mute', 0, 1), 0)
    assert buffer.activity.active(1) == {'speaker_1'}
    buffer.read(320, 'mute', 0, 1.02)
    assert buffer.activity.active(1.02) == {'speaker_1', 'speaker_2'}
    assert buffer.activity.active(1.31) == {'speaker_2'}
    assert not buffer.activity.active(1.33)  # also expires if output callbacks stop
    buffer.read(320, 'all', 1, 2)
    assert not buffer.activity.active(2)


def test_speaking_activity_ignores_low_energy_and_unknown_audio():
    packet = AudioPacket(np.ones(320, np.float32), {'speaker_1': np.full(320, .001, np.float32)})
    assert not packet.active(0, 320)
    assert not packet.active(0, 0)


def test_resampling_preserves_duration():
    wave = np.sin(np.arange(48000) * 2 * np.pi * 440 / 48000).astype(np.float32)
    output = resample(wave, 48000, 16000)
    assert len(output) == 16000
    assert output.dtype == np.float32


def test_unstable_gray_zone_never_promotes_from_duration_alone():
    registry = SpeakerRegistry()
    registry.assign([[1, 0, 0]], [1.2])
    radius = np.sqrt(1 - .35 ** 2)
    variants = [[.35, radius, 0], [.35, radius * .6, radius * .8]]
    for i in range(12):
        assert registry.assign([variants[i % 2]], [1.2]) == [None]
    assert len(registry.speakers) == 1


def test_repeated_subsecond_evidence_registers_without_single_fragment_registration():
    registry = SpeakerRegistry()
    registry.assign([[1, 0]], [1.2])
    assert registry.assign([[0, 1]], [.8]) == [None]
    assert registry.assign([[0, 1]], [.8]) == [None]
    assert registry.assign([[0, 1]], [.8]) == ['speaker_2']


def test_gray_candidates_do_not_block_known_voice_return():
    registry = SpeakerRegistry()
    registry.assign([[1, 0]], [1.2])
    assert registry.assign([[.35, np.sqrt(1-.35**2)]], [1.2]) == [None]
    # Similar to the candidate but still positively identifiable as the original.
    assert registry.assign([[.65, np.sqrt(1-.65**2)]], [1.2]) == ['speaker_1']
    assert not registry.candidates
