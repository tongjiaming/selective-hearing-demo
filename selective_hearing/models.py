"""Real pretrained waveform separation + speaker embedding baseline.

This is NOT diart or WeSep. The two outputs of Conv-TasNet are matched using
ECAPA embeddings. It is a non-causal, windowed demo, not low-latency TSE.
"""
from pathlib import Path
import hashlib
import time

import numpy as np

from .core import AudioPacket, SpeakerRegistry, resample

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def calibrate_sources(estimated, mixture):
    """Resolve scale/sign ambiguity using only the input mixture."""
    estimated = np.asarray(estimated, np.float64)
    mixture = np.asarray(mixture, np.float64)
    if not np.isfinite(estimated).all() or not np.isfinite(mixture).all():
        raise ValueError("模型输出包含无效数值")
    coefficients = np.linalg.lstsq(estimated.T, mixture, rcond=1e-5)[0]
    return (estimated * coefficients[:, None]).astype(np.float32)


class ModelBackend:
    def __init__(self, model_dir=MODEL_DIR, threads=4):
        import torch
        import torchaudio
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from silero_vad import load_silero_vad, get_speech_timestamps
        from .model_assets import ASSETS

        self.torch = torch
        torch.set_num_threads(threads)
        model_dir = Path(model_dir)
        for name, asset in ASSETS.items():
            path = model_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"缺少模型 {name}。请运行 scripts/download_models.py")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if asset.get("sha256") and actual != asset["sha256"]:
                raise ValueError(f"模型校验失败：{name}；请重新下载")
        self.separator = torchaudio.models.conv_tasnet_base(num_sources=2)
        self.separator.load_state_dict(torch.load(model_dir / "conv_tasnet.pt", map_location="cpu", weights_only=True), strict=True)
        self.separator.eval()
        self.features = Fbank(n_mels=80)
        self.normalize = InputNormalization(norm_type="sentence", std_norm=False)
        self.encoder = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
                                 kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                                 attention_channels=128, lin_neurons=192)
        self.encoder.load_state_dict(torch.load(model_dir / "ecapa.ckpt", map_location="cpu", weights_only=True), strict=True)
        self.encoder.eval()
        self.vad = load_silero_vad()
        self.timestamps = get_speech_timestamps

    def embedding(self, audio16):
        torch = self.torch
        with torch.inference_mode():
            signal = torch.from_numpy(np.asarray(audio16, np.float32)).unsqueeze(0)
            features = self.normalize(self.features(signal), torch.ones(1))
            return self.encoder(features).squeeze().cpu().numpy()

    def separate(self, audio16):
        torch = self.torch
        # The pretrained separator is an 8 kHz / two-speaker model.
        signal = resample(audio16, 16000, 8000)
        with torch.inference_mode():
            output = self.separator(torch.from_numpy(signal).reshape(1, 1, -1))[0].cpu().numpy()
        # SI-SDR-trained estimates have arbitrary gain (this checkpoint can amplify
        # by orders of magnitude). Fit each estimated source's gain to the MIXTURE,
        # never to clean references; do not add the residual back to either source.
        output = calibrate_sources(output, signal)
        return [resample(track, 8000, 16000)[:len(audio16)] for track in output]

    def speech(self, audio16):
        torch = self.torch
        with torch.inference_mode():
            self.vad.reset_states()
            segments = self.timestamps(torch.from_numpy(np.asarray(audio16, np.float32)), self.vad,
                                       sampling_rate=16000, threshold=0.6,
                                       min_speech_duration_ms=200)
        if not segments:
            return np.empty(0, np.float32)
        return np.concatenate([audio16[s["start"]:s["end"]] for s in segments])


class Processor:
    def __init__(self, backend, registry=None):
        self.backend = backend
        self.registry = registry if registry is not None else SpeakerRegistry()

    def process(self, audio16, captured_at=0.0):
        begin = time.perf_counter()
        audio16 = np.asarray(audio16, np.float32)
        tracks, vectors, durations = [], [], []
        mixture_rms = float(np.sqrt(np.mean(audio16 ** 2))) if len(audio16) else 0.0
        if len(audio16) >= 16000 and mixture_rms > 0.002:
            separated = self.backend.separate(audio16)
            for track in separated:
                track_rms = float(np.sqrt(np.mean(track ** 2)))
                if track_rms < max(0.002, mixture_rms * 0.12):
                    continue
                speech = self.backend.speech(track)
                if len(speech) < 12000:
                    continue
                vectors.append(self.backend.embedding(speech))
                durations.append(len(speech) / 16000)
                tracks.append(track)
        ids = self.registry.assign(vectors, durations)
        mapped = {identity: track for identity, track in zip(ids, tracks) if identity is not None}
        elapsed = time.perf_counter() - begin
        duration = len(audio16) / 16000
        return AudioPacket(audio16, mapped, captured_at), {
            "seconds": duration, "compute_seconds": elapsed,
            "rtf": elapsed / max(duration, 0.001),
            "active": [s for s in ids if s is not None],
            "unmatched": sum(s is None for s in ids),
            "pending_speakers": len(self.registry.candidates),
            "speakers": self.registry.snapshot(),
        }
