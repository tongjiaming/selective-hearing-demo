import numpy as np
import pytest

from selective_hearing.core import SpeakerRegistry, AudioPacket, PlaybackBuffer, resample, unit


def test_output_permutation_does_not_swap_identity():
    registry = SpeakerRegistry()
    a, b = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    assert registry.assign([a, b], [1., 1.]) == ["speaker_1", "speaker_2"]
    assert not any(s["ready"] for s in registry.snapshot())
    assert registry.assign([b, a], [1., 1.]) == ["speaker_2", "speaker_1"]
    assert all(s["ready"] for s in registry.snapshot())


def test_unknown_voice_never_forced_to_selected_identity():
    registry = SpeakerRegistry()
    registry.assign([[1, 0, 0], [0, 1, 0]], [2, 2])
    assert registry.assign([[0, 0, 1]], [2]) == [None]
    assert len(registry.speakers) == 2


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
