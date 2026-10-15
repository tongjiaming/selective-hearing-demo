"""Optional full diart evaluation, isolated from live audio and saved identities.

This uses diart's actual segmentation/embedding/online-clustering pipeline.
Model authorization and real-weight evaluation are required before enabling it
as a live backend. No microphone, playback device or daily profile is opened.
"""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import time

from .profiles import validate_window_seconds

SEGMENTATION = 'pyannote/segmentation-3.0'
EMBEDDING = 'speechbrain/spkrec-ecapa-voxceleb'


def check_environment():
    import diart  # Check real imports, not just installed package metadata.
    from huggingface_hub import get_token
    return {'diart': version('diart'), 'pyannote.audio': version('pyannote.audio'),
            'torch': version('torch'), 'imports_ok': True,
            'hf_login_available': bool(get_token()), 'real_weights_validated': False,
            'segmentation': SEGMENTATION, 'embedding': EMBEDDING}


def build_pipeline(duration=5, step=.5, latency=1.5):
    import math
    from huggingface_hub import get_token
    duration = validate_window_seconds(duration)
    if not all(math.isfinite(x) for x in (step, latency)) or not 0 < step <= latency <= duration:
        raise ValueError('diart 参数需满足 0 < step <= latency <= duration。')
    if not get_token():
        raise RuntimeError('尚未登录 Hugging Face。请先在 https://huggingface.co/pyannote/segmentation-3.0 '
                           '接受模型条件，再在本机运行 huggingface-cli login；不要把 token 发到聊天或提交到仓库。')
    import torch
    from diart import SpeakerDiarization, SpeakerDiarizationConfig
    from diart.models import SegmentationModel, EmbeddingModel
    torch.set_num_threads(4)
    config = SpeakerDiarizationConfig(
        segmentation=SegmentationModel.from_pretrained(SEGMENTATION, use_hf_token=True),
        embedding=EmbeddingModel.from_pretrained(EMBEDDING, use_hf_token=True),
        duration=duration, step=step, latency=latency, device=torch.device('cpu'),
        sample_rate=16000)
    return SpeakerDiarization(config)


def run_file(path, pipeline, output):
    from diart.sources import FileAudioSource
    from diart.inference import StreamingInference
    from pyannote.core import Segment
    import soundfile as sf

    path, output = Path(path), Path(output)
    info = sf.info(path)
    if not 0 < info.duration <= 600:
        raise ValueError('测试音频需为 0 到 600 秒。')
    pipeline.reset()
    padding = pipeline.config.get_file_padding(path)
    source = FileAudioSource(path, 16000, padding, pipeline.config.step)
    pipeline.set_timestamp_shift(-padding[0])

    class CheckedInference(StreamingInference):
        error = None

        def _handle_error(self, error):
            # Upstream prints and swallows inference errors; do not export a
            # partial result as a successful run or print model access tokens.
            self.error = error
            self.source.close()

    inference = CheckedInference(pipeline, source, do_plot=False,
                                 do_profile=False, show_progress=False)
    begin = time.perf_counter()
    try:
        prediction = inference()
    finally:
        source.close()
    if inference.error is not None:
        raise RuntimeError('diart 推理失败：' + type(inference.error).__name__) from None
    elapsed = time.perf_counter() - begin
    prediction = prediction.crop(Segment(0, info.duration))
    segments = [{'start': round(segment.start, 4), 'end': round(segment.end, 4), 'speaker': label}
                for segment, _, label in prediction.itertracks(yield_label=True)]
    report = {'backend': 'diart', 'duration_seconds': info.duration,
              'analysis_window_seconds': pipeline.config.duration,
              'step_seconds': pipeline.config.step, 'algorithm_latency_seconds': pipeline.config.latency,
              'compute_seconds': elapsed, 'rtf': elapsed / info.duration,
              'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'speakers': prediction.labels(), 'segments': segments,
              'note': '身份与时间段，不是分离音轨；匿名标签仅本次测试有效，不修改日常人物库。'
                      'RTF 为文件回放计算时间，不是设备端到端延迟，也不代表识别准确率。'}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'diarization.rttm').open('w', encoding='utf-8') as stream:
        prediction.write_rttm(stream)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
