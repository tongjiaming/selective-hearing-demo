"""Explicitly started input/output; inference never runs in audio callbacks."""
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Thread
import json
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from .core import AudioPacket, PlaybackBuffer, mono, resample
from .models import Processor

LIVE_WINDOW_SECONDS = 1.5


def stream_settings(device):
    """Use the same format conversion for probing and opening Windows endpoints."""
    info = sd.query_devices(device)
    api = sd.query_hostapis(info['hostapi'])
    if api['name'] == 'Windows WASAPI':
        return sd.WasapiSettings(auto_convert=True)
    return None


def devices(refresh=False):
    if refresh:
        # PortAudio caches the device list. Only call this with all streams closed.
        # Private hooks are isolated here and sounddevice is pinned to 0.5.3.
        if sd._initialized:
            sd._terminate()
        sd._initialize()
    apis = sd.query_hostapis()
    # WASAPI exposes active endpoints once, without legacy mapper aliases or KS pins.
    api_id = next((i for i, api in enumerate(apis) if api['name'] == 'Windows WASAPI'),
                  sd.default.hostapi)
    api = apis[api_id]
    result = []
    for i, device in enumerate(sd.query_devices()):
        if device['hostapi'] != api_id:
            continue
        item = {'id': i, 'name': device['name'], 'api': api['name'],
                'rate': int(device['default_samplerate'])}
        for direction, check in [('input', sd.check_input_settings),
                                 ('output', sd.check_output_settings)]:
            item[direction] = 0
            item[f'default_{direction}'] = i == api[f'default_{direction}_device']
            if not device[f'max_{direction}_channels']:
                continue
            try:
                # This queries capabilities; it never opens or starts a microphone.
                check(device=i, channels=1, samplerate=16000, dtype='float32',
                      extra_settings=stream_settings(i))
            except (sd.PortAudioError, ValueError):
                continue
            item[direction] = device[f'max_{direction}_channels']
        if item['input'] or item['output']:
            result.append(item)
    return result


def load_audio(path):
    signal, rate = sf.read(path, dtype="float32", always_2d=True)
    signal = resample(mono(signal), rate, 16000)
    if not len(signal):
        raise ValueError("音频文件为空")
    return signal


def process_file(path, backend, notify=lambda _: None, cancel=None, window_seconds=4):
    audio = load_audio(path)
    if len(audio) > 16000 * 600:
        raise ValueError("首版单文件最多 10 分钟，请先截取测试片段")
    processor = Processor(backend)
    tracks, metrics = {}, []
    size = int(window_seconds * 16000)
    for start in range(0, len(audio), size):
        if cancel and cancel.is_set():
            raise InterruptedError("已取消文件处理")
        part = audio[start:start + size]
        packet, info = processor.process(part)
        for identity, track in packet.tracks.items():
            if identity not in tracks:
                tracks[identity] = np.zeros_like(audio)
            tracks[identity][start:start + len(part)] = track
        info["offset_seconds"] = start / 16000
        metrics.append(info)
        notify({"type": "progress", "fraction": min(1, (start + size) / len(audio)), **info})
    # Provisional identities can already be selected; confidence stays in the report.
    return AudioPacket(audio, tracks), {
        "sample_rate": 16000, "duration_seconds": len(audio) / 16000,
        "window_seconds": window_seconds, "speakers": processor.registry.snapshot(),
        "chunks": metrics, "compute_seconds": sum(x["compute_seconds"] for x in metrics),
        "backend": "Conv-TasNet Libri2Mix + ECAPA-TDNN + Silero VAD",
        "note": "分块非因果处理。指标不包含蓝牙/设备延迟。未匹配的目标片段输出静音。",
    }


def export_result(packet, report, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / "mixture.wav", packet.original, 16000)
    for identity, track in packet.tracks.items():
        sf.write(folder / f"{identity}.wav", np.clip(track, -0.95, 0.95), 16000)
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


class FilePlayer:
    def __init__(self, packet, device, control, loop=True):
        self.packet, self.control, self.loop = packet, control, loop
        self.offset = 0
        self.finished = False
        self.stream = sd.OutputStream(device=device, samplerate=16000, channels=1,
                                      dtype="float32", blocksize=320, callback=self._callback,
                                      extra_settings=stream_settings(device))

    def _callback(self, outdata, frames, timing, status):
        mode, gain = self.control.get()
        length = len(self.packet.original)
        data = np.zeros(frames, np.float32)
        written = 0
        while written < frames:
            remaining = length - self.offset
            if remaining == 0:
                if not self.loop or not length:
                    self.finished = True
                    break
                self.offset = 0
                remaining = length
            count = min(frames - written, remaining)
            data[written:written + count] = self.packet.slice(mode, self.offset, self.offset + count)
            written += count
            self.offset += count
        outdata[:, 0] = np.clip(data * gain, -0.95, 0.95)

    def start(self):
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None


class LiveSession:
    def __init__(self, input_device, output_device, control, notify, backend=None,
                 window_seconds=LIVE_WINDOW_SECONDS):
        self.input_device, self.output_device = input_device, output_device
        self.control, self.notify = control, notify
        self.backend = backend
        self.processor = Processor(backend) if backend else None
        self.window = int(window_seconds * 16000)
        self.cancel = Event()
        # Capture chunks are 20 ms; max queue is one model window, no indefinite buildup.
        self.capture = Queue(maxsize=int(window_seconds * 50))
        self.playback = PlaybackBuffer(max_samples=self.window * 2 if backend else 3200)
        self.input_drops = 0
        self.input_overflows = 0
        self.output_underflows = 0
        self.input_stream = self.output_stream = None
        self.worker = None
        self.level = 0.0
        self.sequence = 0

    def start(self):
        input_settings = stream_settings(self.input_device)
        output_settings = stream_settings(self.output_device)
        sd.check_input_settings(device=self.input_device, channels=1, samplerate=16000,
                                dtype="float32", extra_settings=input_settings)
        sd.check_output_settings(device=self.output_device, channels=1, samplerate=16000,
                                 dtype="float32", extra_settings=output_settings)
        try:
            self.output_stream = sd.OutputStream(device=self.output_device, samplerate=16000,
                channels=1, dtype="float32", blocksize=320, callback=self._output,
                extra_settings=output_settings)
            self.input_stream = sd.InputStream(device=self.input_device, samplerate=16000,
                channels=1, dtype="float32", blocksize=320, callback=self._input,
                extra_settings=input_settings)
            self.worker = Thread(target=self._work, daemon=True)
            self.worker.start()
            self.output_stream.start()
            self.input_stream.start()
        except Exception:
            self.stop()
            raise

    def _input(self, indata, frames, timing, status):
        if status.input_overflow:
            self.input_overflows += 1
        audio = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(audio ** 2)))
        self.sequence += 1
        item = (self.sequence, time.perf_counter(), audio)
        if self.processor is None:
            self.playback.put(AudioPacket(audio, captured_at=item[1]))
            return
        try:
            self.capture.put_nowait(item)
        except Full:
            try:
                self.capture.get_nowait()
            except Empty:
                pass
            self.input_drops += 1
            try:
                self.capture.put_nowait(item)
            except Full:
                self.input_drops += 1

    def _output(self, outdata, frames, timing, status):
        if status.output_underflow:
            self.output_underflows += 1
        mode, gain = self.control.get()
        outdata[:, 0] = self.playback.read(frames, mode, gain, time.perf_counter())

    def _work(self):
        chunks, count, last_sequence, captured_at = [], 0, None, 0.0
        while not self.cancel.is_set():
            try:
                sequence, stamp, audio = self.capture.get(timeout=0.1)
            except Empty:
                continue
            if last_sequence is not None and sequence != last_sequence + 1:
                # Never concatenate across a capture discontinuity without acknowledging it.
                chunks, count = [], 0
            last_sequence = sequence
            if not chunks:
                captured_at = stamp
            chunks.append(audio)
            count += len(audio)
            if count < self.window:
                continue
            mixture = np.concatenate(chunks)
            chunks, count = [], 0
            try:
                packet, info = self.processor.process(mixture, captured_at)
            except Exception as exc:
                self.notify({"type": "error", "message": str(exc)})
                self.cancel.set()
                return
            if not self.cancel.is_set():
                self.playback.put(packet)
                self.notify({"type": "progress", **info})

    def stats(self):
        return {"input_drops": self.input_drops, "input_overflows": self.input_overflows,
                "output_underflows": self.output_underflows, "output_drops": self.playback.dropped,
                "empty_callbacks": self.playback.underruns, "level": self.level,
                "software_age": self.playback.last_age, "queue_seconds": self.playback.samples / 16000}

    def stop(self):
        self.cancel.set()
        for stream in (self.input_stream, self.output_stream):
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        self.input_stream = self.output_stream = None
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=0.2)
