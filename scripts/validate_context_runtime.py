"""Measure actual processing plus SQLite learning on public audio, without devices."""
import json
from pathlib import Path
import tempfile

import numpy as np

from validate_identity_classification import ROOT, fixtures, ModelBackend, Processor, SpeakerRegistry
from selective_hearing.profiles import ProfileStore


def main():
    backend = ModelBackend()
    _, waveform, _, window = fixtures()[0][1]  # Public overlapping pair, 1.5 s.
    size = int(window * 16000)
    # Warm up the model only; each measured registry starts empty.
    Processor(backend).process(waveform[:size])
    measurements = {}
    for context in (0, 3):
        with tempfile.TemporaryDirectory(prefix='selective-hearing-public-') as folder:
            registry = SpeakerRegistry(store=ProfileStore(Path(folder) / 'public.sqlite3'))
            processor = Processor(backend, registry, context_seconds=context)
            blocks = []
            for start in range(0, len(waveform) - size + 1, size):
                _, report = processor.process(waveform[start:start+size])
                blocks.append(report['compute_seconds'])
            measurements[str(context)] = dict(
                full_blocks=len(blocks), window_seconds=window,
                median_compute_seconds=float(np.median(blocks)),
                p95_compute_seconds=float(np.percentile(blocks, 95)),
                max_compute_seconds=max(blocks),
                blocks_slower_than_capture=sum(seconds > window for seconds in blocks),
                compute_seconds=blocks, people=len(registry.speakers))
            print(context, measurements[str(context)], flush=True)
    report = dict(method='Local CPU, warmed model, actual Processor and temporary SQLite profiles. Public overlapping speech only; excludes audio device and Bluetooth latency. Single run, not a sustained-load guarantee.',
                  contexts=measurements)
    (ROOT / 'docs/evidence/context-runtime.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
