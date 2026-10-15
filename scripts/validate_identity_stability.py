"""Compare identity tracking on identical real model outputs; no microphone access.

Public clean references are only used to construct fixtures and score results.
The model only receives each mixed waveform. No embeddings or audio are exported.
"""
import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selective_hearing.audio import load_audio
from selective_hearing.models import ModelBackend, Processor
from validate_models import si_sdr


def baseline_registry(ref):
    source = subprocess.check_output(['git', 'show', f'{ref}:selective_hearing/core.py'], cwd=ROOT)
    module = types.ModuleType('_baseline_identity_core')
    sys.modules[module.__name__] = module
    exec(compile(source, f'{ref}/core.py', 'exec'), module.__dict__)
    return module.SpeakerRegistry


def evaluate(backend, old_class, waveform, references, window_seconds):
    old = old_class()
    processor = Processor(backend)
    current_assign = processor.registry.assign
    before_trace, after_trace = [], []

    def compare(vectors, durations):
        before_trace.append(old.assign(vectors, durations))
        assigned = current_assign(vectors, durations)
        after_trace.append(assigned)
        return assigned

    processor.registry.assign = compare
    tracks = {}
    size = int(16000 * window_seconds)
    for start in range(0, len(waveform), size):
        packet, _ = processor.process(waveform[start:start + size])
        for identity, track in packet.tracks.items():
            tracks.setdefault(identity, np.zeros_like(waveform))[start:start + len(track)] = track
    result = {'expected_people': len(references), 'before_people': len(old.speakers),
              'after_people': len(processor.registry.speakers), 'window_seconds': window_seconds,
              'duration_seconds': len(waveform) / 16000,
              'input_sha256': hashlib.sha256(waveform.tobytes()).hexdigest(),
              'eligible_tracks': sum(map(len, after_trace)),
              'before_matched_tracks': sum(x is not None for row in before_trace for x in row),
              'after_matched_tracks': sum(x is not None for row in after_trace for x in row),
              'before_assignments': before_trace, 'after_assignments': after_trace}
    if len(tracks) == len(references):
        order = max(permutations(tracks), key=lambda ids: sum(si_sdr(r, tracks[k]) for r, k in zip(references, ids)))
        result['global_identity_si_sdr_db'] = [round(si_sdr(r, tracks[k]), 3) for r, k in zip(references, order)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='21200ae')
    parser.add_argument('--output', default='docs/evidence/identity-stability.json')
    args = parser.parse_args()
    references, manifest = [], []
    for speaker in (1, 2):
        parts = []
        for sentence in range(1, 7):
            path = ROOT / 'work/samples' / f'spk{speaker}_snt{sentence}.wav'
            parts.extend([load_audio(path), np.zeros(2400, np.float32)])
            manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        wave = np.concatenate(parts)
        references.append(wave * (0.08 / max(float(np.sqrt(np.mean(wave ** 2))), 1e-8)))
    length = max(map(len, references))
    a, b = [np.pad(w, (0, length - len(w))) for w in references]
    silence = np.zeros(64000, np.float32)
    arrival = [np.concatenate([a[:64000], silence, a[64000:]]), np.concatenate([silence, b])]
    rng = np.random.default_rng(20261015)
    fixtures = [('single_live', a, [a], 1.5), ('overlap_live', a + b, [a, b], 1.5),
                ('arrival_live', sum(arrival), arrival, 1.5),
                ('quiet_noisy_overlap_live', (a + b) * .3 + rng.normal(0, .001, len(a)).astype(np.float32),
                 [a * .3, b * .3], 1.5), ('overlap_file', a + b, [a, b], 4)]
    old_class = baseline_registry(args.baseline_ref)
    backend = ModelBackend()
    cases = {}
    for name, waveform, refs, window in fixtures:
        cases[name] = evaluate(backend, old_class, waveform, refs, window)
        print(name, json.dumps({k: v for k, v in cases[name].items() if not k.endswith('assignments')}, ensure_ascii=False), flush=True)
    report = {'baseline_ref': args.baseline_ref, 'source': 'SpeechBrain public ASR samples',
              'method': 'Same separated embeddings sent to old and new registries. Live windows are non-overlapping 1.5 s.',
              'limitations': 'Small synthetic English fixtures; not a user microphone recording. Unknown segments remain silent; no guarantee of real-room accuracy.',
              'manifest': manifest, 'cases': cases}
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert all(r['after_people'] == r['expected_people'] for r in cases.values()), 'Unexpected identity count; inspect evidence'


if __name__ == '__main__':
    main()
