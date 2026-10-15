"""Diart integration plumbing only; synthetic models are not accuracy evidence."""
import numpy as np
import pytest
import soundfile as sf

from selective_hearing.diart_probe import build_pipeline, run_file


def test_missing_login_blocks_model_loading_without_secret_input(monkeypatch):
    monkeypatch.setattr('huggingface_hub.get_token', lambda: None)
    with pytest.raises(RuntimeError, match='尚未登录'):
        build_pipeline()


def synthetic_pipeline():
    pytest.importorskip('diart')
    import torch
    from diart import SpeakerDiarization, SpeakerDiarizationConfig
    from diart.models import SegmentationModel, EmbeddingModel
    class Segmentation(torch.nn.Module):
        def forward(self, waveform):
            return torch.full((waveform.shape[0], 100, 2), .9)
    class Embedding(torch.nn.Module):
        def forward(self, waveform, weights=None):
            return torch.ones((waveform.shape[0], 192))
    return SpeakerDiarization(SpeakerDiarizationConfig(
        segmentation=SegmentationModel(Segmentation), embedding=EmbeddingModel(Embedding),
        duration=2, step=.5, latency=.5, device=torch.device('cpu')))


def test_actual_diart_stream_exports_bounded_annotation_without_audio_devices(tmp_path, monkeypatch):
    pipeline = synthetic_pipeline()
    import sounddevice
    monkeypatch.setattr(sounddevice, 'InputStream', lambda **kw: pytest.fail('opened mic'))
    monkeypatch.setattr(sounddevice, 'OutputStream', lambda **kw: pytest.fail('opened playback'))
    path = tmp_path / 'synthetic.wav'
    sf.write(path, np.full(32000, .1, np.float32), 16000)
    report = run_file(path, pipeline, tmp_path / 'out')
    assert report['backend'] == 'diart' and report['segments']
    assert report['duration_seconds'] == 2
    assert all(0 <= s['start'] < s['end'] <= 2 for s in report['segments'])
    assert (tmp_path / 'out/diarization.rttm').is_file()
    assert (tmp_path / 'out/report.json').is_file()


def test_upstream_swallowed_inference_error_is_not_reported_as_success(tmp_path):
    pipeline = synthetic_pipeline()
    def fail(_):
        raise ValueError('synthetic inference failure')
    pipeline.segmentation = fail
    path = tmp_path / 'synthetic.wav'
    sf.write(path, np.full(32000, .1, np.float32), 16000)
    with pytest.raises(RuntimeError, match='diart 推理失败'):
        run_file(path, pipeline, tmp_path / 'out')
    assert not (tmp_path / 'out/report.json').exists()
