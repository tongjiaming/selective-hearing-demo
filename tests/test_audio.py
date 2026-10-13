from queue import Queue
from threading import Event

import numpy as np
import soundfile as sf
import pytest

from selective_hearing.audio import process_file, export_result, FilePlayer, LiveSession
from selective_hearing.core import AudioPacket, OutputControl


class FakeBackend:
    """Deterministic tests of plumbing ONLY; never shipped as an app backend."""
    def __init__(self):
        self.flip = False

    def separate(self, audio):
        self.flip = not self.flip
        tracks = [np.ones_like(audio) * .1, np.ones_like(audio) * -.1]
        return tracks if self.flip else tracks[::-1]

    def speech(self, audio):
        return audio

    def embedding(self, audio):
        return np.array([1, 0]) if np.mean(audio) > 0 else np.array([0, 1])


def test_file_tracks_stay_aligned_across_permutation(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.ones(16000 * 8, np.float32) * .15, 16000)
    packet, report = process_file(source, FakeBackend())
    assert len(packet.tracks) == 2
    np.testing.assert_allclose(packet.tracks["speaker_1"], .1)
    np.testing.assert_allclose(packet.tracks["speaker_2"], -.1)
    folder = tmp_path / "result"
    export_result(packet, report, folder)
    assert (folder / "report.json").exists()
    assert sf.info(folder / "speaker_1.wav").frames == 128000


def test_cancel_and_short_unconfirmed_file(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.ones(16000 * 2, np.float32) * .15, 16000)
    packet, report = process_file(source, FakeBackend())
    assert len(packet.tracks) == 0  # one observation is insufficient
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        process_file(source, FakeBackend(), cancel=cancel)


def test_file_callback_switch_keeps_position_without_device():
    player = object.__new__(FilePlayer)
    player.packet = AudioPacket(np.ones(10, np.float32) * .2,
                                {"speaker_1": np.arange(10, dtype=np.float32) / 20})
    player.control = OutputControl()
    player.control.set(gain=1)
    player.offset, player.loop, player.finished = 0, True, False
    data = np.empty((3, 1), np.float32)
    player._callback(data, 3, None, None)
    player.control.set(mode="speaker_1")
    player._callback(data, 3, None, None)
    np.testing.assert_allclose(data[:, 0], [.15, .2, .25])


def test_live_stop_is_idempotent_without_devices():
    session = LiveSession(0, 1, OutputControl(), lambda _: None)
    session.stop()
    session.stop()
    assert session.cancel.is_set()
