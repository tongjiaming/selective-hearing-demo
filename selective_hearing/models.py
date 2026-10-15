"""Real pretrained waveform separation + speaker embedding baseline.

This is NOT diart or WeSep. The two outputs of Conv-TasNet are matched using
ECAPA embeddings. It is a non-causal, windowed demo, not low-latency TSE.
"""
from pathlib import Path
import hashlib
import time

import numpy as np

from .core import AudioPacket, SpeakerRegistry, resample
from .feature_bank import MIN_SPEECH_SECONDS

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
        # The packaged ONNX VAD runs on its own single-threaded CPU session.
        # Repeated tiny Torch VAD calls otherwise consume a large share of the
        # short-window budget. No network or additional weights are required.
        self.vad = load_silero_vad(onnx=True)
        self.timestamps = get_speech_timestamps
        # Build CPU kernels for the rolling context while the UI still says
        # loading, rather than stalling the second live block. Synthetic zeros
        # never reach identity learning, playback, or any persistent store.
        warmup = np.zeros(48000, np.float32)
        self.separate(warmup)
        self.embedding(warmup)

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

    def speech_mask(self, audio16):
        torch = self.torch
        with torch.inference_mode():
            self.vad.reset_states()
            segments = self.timestamps(torch.from_numpy(np.asarray(audio16, np.float32)), self.vad,
                                       sampling_rate=16000, threshold=0.6,
                                       min_speech_duration_ms=200)
        mask = np.zeros(len(audio16), dtype=bool)
        for segment in segments:
            mask[segment['start']:segment['end']] = True
        return mask

    def speech(self, audio16):
        return audio16[self.speech_mask(audio16)]


class Processor:
    def __init__(self, backend, registry=None, context_seconds=3.0):
        self.backend = backend
        self.registry = registry if registry is not None else SpeakerRegistry()
        # A backend without frame-aligned VAD cannot safely distinguish old
        # speech from current activity, so it retains the independent windows.
        self.context_samples = int(context_seconds * 16000) if hasattr(backend, 'speech_mask') else 0
        self.history = np.empty(0, np.float32)
        self.position = 0
        self.reference_end = 0
        self.reference_eligible = True

    def extract_features(self, audio16):
        """Past-only context; return audio/activity strictly for the NEW block."""
        audio16 = np.asarray(audio16, np.float32)
        context = np.concatenate([self.history, audio16]) if self.context_samples else audio16
        context = context[-max(self.context_samples, len(audio16)):] if len(context) else context
        self.position += len(audio16)
        # Retain only transient RAM audio, never write context to profiles.
        self.history = context[-self.context_samples:].copy() if self.context_samples else np.empty(0, np.float32)
        self.reference_eligible = self.position - len(context) >= self.reference_end
        tracks, vectors, durations = [], [], []
        voiced_seconds = []
        mixture_rms = float(np.sqrt(np.mean(audio16 ** 2))) if len(audio16) else 0.0
        minimum_samples = int(MIN_SPEECH_SECONDS * 16000)
        if len(audio16) >= minimum_samples and mixture_rms > 0.002:
            separated = self.backend.separate(context)
            for full_track in separated:
                track = full_track[-len(audio16):]
                track_rms = float(np.sqrt(np.mean(track ** 2)))
                if track_rms < max(0.002, mixture_rms * 0.12):
                    continue
                if self.context_samples:
                    mask = self.backend.speech_mask(full_track)
                    speech = full_track[mask]
                    current_samples = int(np.count_nonzero(mask[-len(audio16):]))
                else:
                    speech = self.backend.speech(track)
                    current_samples = len(speech)
                voiced_seconds.append(current_samples / 16000)
                if current_samples < minimum_samples:
                    continue
                vectors.append(self.backend.embedding(speech))
                durations.append(current_samples / 16000)
                tracks.append(track)
        if vectors and self.reference_eligible:
            self.reference_end = self.position
        return tracks, vectors, durations, voiced_seconds

    def process(self, audio16, captured_at=0.0):
        begin = time.perf_counter()
        audio16 = np.asarray(audio16, np.float32)
        tracks, vectors, durations, voiced_seconds = self.extract_features(audio16)
        mixture_rms = float(np.sqrt(np.mean(audio16 ** 2))) if len(audio16) else 0.0
        minimum_samples = int(MIN_SPEECH_SECONDS * 16000)
        ids = self.registry.assign(vectors, durations, reference_eligible=self.reference_eligible)
        mapped = {}
        for identity, track in zip(ids, tracks):
            if identity is not None:
                mapped[identity] = mapped.get(identity, np.zeros_like(track)) + track
        decisions = self.registry.decisions
        messages = []
        created = sum(d['status'] == 'created' for d in decisions)
        recognized = len({d['speaker_id'] for d in decisions if d['status'] == 'recognized'} -
                         {d.get('speaker_id') for d in decisions if d['status'] == 'created'})
        if created:
            messages.append(f'已新增 {created} 人（先建档，继续确认）')
        if recognized:
            messages.append(f'匹配到已有 {recognized} 人')
        for decision in decisions:
            if decision['status'] == 'confirming':
                messages.append(f"新声音确认中：稳定证据 {decision['observations']}/{decision['required_observations']}，"
                                f"有效语音 {decision['seconds']:.1f}/{decision['required_seconds']:.1f} 秒")
                if decision['observations'] < decision['required_observations']:
                    if decision.get('collected_observations', 1) > decision['observations']:
                        messages.append('已收到多段人声，但声纹一致性不足；可尝试加长分析窗口')
                    elif decision['observations'] == 1:
                        messages.append('已检出人声，等待相似的跨窗片段')
            elif decision['status'].startswith('ambiguous'):
                messages.append('已检出人声，归属待确认；特征已暂存，并非已确认的背景音')
            elif decision['status'] == 'short_speech':
                messages.append('有效人声太短，暂不建档')
        if not messages:
            if mixture_rms <= .002:
                messages.append('输入过弱，尚未进入人声识别；请检查麦克风输入音量')
            elif len(audio16) < minimum_samples:
                messages.append(f'本段不足 {MIN_SPEECH_SECONDS:g} 秒，未参与声纹识别')
            else:
                messages.append(f'本段有效人声最长 {max(voiced_seconds, default=0):.2f} 秒，未达到 {MIN_SPEECH_SECONDS:g} 秒')
        if self.registry.last_corrections:
            messages.append(f'更新 {len(self.registry.last_corrections)} 段历史特征归属')
        if self.registry.last_merges:
            messages.append(f'已合并 {len(self.registry.last_merges)} 个证据不足的重复人物')
        elapsed = time.perf_counter() - begin
        duration = len(audio16) / 16000
        return AudioPacket(audio16, mapped, captured_at), {
            "seconds": duration, "compute_seconds": elapsed,
            "rtf": elapsed / max(duration, 0.001),
            "active": list(dict.fromkeys(s for s in ids if s is not None)),
            "unmatched": sum(s is None for s in ids),
            "detected_speech_tracks": sum(seconds > 0 for seconds in voiced_seconds),
            "short_speech_tracks": sum(MIN_SPEECH_SECONDS <= seconds < .5 for seconds in durations),
            "pending_speakers": len(self.registry.candidates),
            "profile_aliases": dict(self.registry.aliases),
            "profile_merges": list(self.registry.last_merges),
            "recognition_status": '；'.join(dict.fromkeys(messages)),
            "identity_decisions": decisions,
            "feature_samples": len(self.registry.feature_bank.samples),
            "reassigned_samples": len(self.registry.last_corrections),
            "speakers": self.registry.snapshot(),
        }
