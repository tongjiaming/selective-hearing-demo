from queue import Queue
from threading import Event, Thread

import numpy as np
import soundfile as sf
import pytest
from types import SimpleNamespace

from selective_hearing.audio import process_file, export_result, FilePlayer, LiveSession
from selective_hearing.core import AudioPacket, OutputControl
from selective_hearing import audio


def test_device_list_uses_active_endpoints_and_filters_each_direction(monkeypatch):
    apis = [dict(name='MME', default_input_device=0, default_output_device=0),
            dict(name='Windows WASAPI', default_input_device=1, default_output_device=2)]
    raw = [dict(name=name, hostapi=api, max_input_channels=ins, max_output_channels=outs,
                default_samplerate=48000) for name, api, ins, outs in [
        ('Mapper', 0, 2, 2), ('Mic', 1, 1, 0), ('Headphones', 1, 0, 2),
        ('Unsupported', 1, 1, 2), ('Duplex', 1, 1, 2), ('Mic', 1, 1, 0)]]
    monkeypatch.setattr(audio.sd, 'query_hostapis', lambda index=None: apis if index is None else apis[index])
    monkeypatch.setattr(audio.sd, 'query_devices', lambda index=None: raw if index is None else raw[index])
    monkeypatch.setattr(audio.sd, 'default', SimpleNamespace(hostapi=0))
    conversion = object()
    monkeypatch.setattr(audio.sd, 'WasapiSettings', lambda **kw: conversion if kw == {'auto_convert': True} else None)
    calls = []
    def check(direction, **kwargs):
        calls.append((direction, kwargs))
        if kwargs['device'] == 3 or (kwargs['device'] == 4 and direction == 'input'):
            raise audio.sd.PortAudioError('Unsupported format')
    monkeypatch.setattr(audio.sd, 'check_input_settings', lambda **kw: check('input', **kw))
    monkeypatch.setattr(audio.sd, 'check_output_settings', lambda **kw: check('output', **kw))
    monkeypatch.setattr(audio.sd, 'InputStream', lambda **kw: pytest.fail('Enumeration opened microphone'))
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: pytest.fail('Enumeration opened playback'))
    result = audio.devices()
    assert [d['id'] for d in result] == [1, 2, 4, 5]  # identical names may be distinct endpoints
    assert result[0]['default_input'] and result[1]['default_output']
    assert result[2]['input'] == 0 and result[2]['output'] == 2
    assert all(kw['extra_settings'] is conversion and kw['samplerate'] == 16000
               and kw['channels'] == 1 and kw['dtype'] == 'float32' for _, kw in calls)


def test_players_use_the_same_conversion_as_device_probe(monkeypatch):
    conversion = object()
    monkeypatch.setattr(audio, 'stream_settings', lambda device: conversion)
    streams, checks = [], []
    class Stream:
        def __init__(self, **kwargs):
            streams.append(kwargs)
        def start(self): pass
        def stop(self): pass
        def close(self): pass
    monkeypatch.setattr(audio.sd, 'InputStream', Stream)
    monkeypatch.setattr(audio.sd, 'OutputStream', Stream)
    monkeypatch.setattr(audio.sd, 'check_input_settings', lambda **kw: checks.append(kw))
    monkeypatch.setattr(audio.sd, 'check_output_settings', lambda **kw: checks.append(kw))
    player = FilePlayer(AudioPacket(np.zeros(320, np.float32)), 2, OutputControl())
    session = LiveSession(1, 2, OutputControl(), lambda _: None)
    try:
        session.start()
        assert len(streams) == 3 and len(checks) == 2
        assert all(kw['extra_settings'] is conversion for kw in streams + checks)
    finally:
        session.stop()
        player.stop()


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
    np.testing.assert_array_equal(packet.tracks["speaker_2"][:64000], 0)  # pending discovery
    np.testing.assert_allclose(packet.tracks["speaker_2"][64000:], -.1)
    folder = tmp_path / "result"
    export_result(packet, report, folder)
    assert (folder / "report.json").exists()
    assert sf.info(folder / "speaker_1.wav").frames == 128000


def test_cancel_and_short_file_keeps_selectable_provisional_tracks(tmp_path):
    source = tmp_path / "input.wav"
    sf.write(source, np.ones(16000 * 2, np.float32) * .15, 16000)
    packet, report = process_file(source, FakeBackend())
    assert len(packet.tracks) == 1  # first person is immediately selectable; later voices need repeat evidence
    assert report['chunks'][0]['pending_speakers'] == 1
    assert not any(s['ready'] for s in report['speakers'])
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        process_file(source, FakeBackend(), cancel=cancel)


def test_live_first_person_event_before_second_observation_without_hardware():
    class SingleSpeaker(FakeBackend):
        def separate(self, audio):
            return [audio, np.zeros_like(audio)]
    events = Queue()
    session = LiveSession(0, 1, OutputControl(), events.put, SingleSpeaker())
    session.worker = Thread(target=session._work, daemon=True)
    session.worker.start()
    try:
        for _ in range(75):
            session._input(np.full((320, 1), .1, np.float32), 320, None,
                           SimpleNamespace(input_overflow=False))
        event = events.get(timeout=3)
        assert event['seconds'] == 1.5
        assert event['active'] == ['speaker_1']
        assert event['speakers'][0]['observations'] == 1
        assert not event['speakers'][0]['ready']
        packet = session.playback.packets[0]
        assert np.any(packet.select('speaker_1'))
        assert not np.any(packet.select('exclude:speaker_1'))
    finally:
        session.stop()


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
