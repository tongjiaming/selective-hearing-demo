"""Compare identities, not just emitted IDs, on public waveform fixtures.

Clean sources are used only for fixture construction and evaluation labels.
No microphone is opened. Only aggregate metrics/IDs/hashes are exported.
"""
import argparse
import base64
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selective_hearing.audio import load_audio
from selective_hearing.core import SpeakerRegistry
from selective_hearing.models import ModelBackend, Processor
from validate_identity_stability import baseline_registry

PUBLIC_COMMIT = '9ef9a3478a9fbc9cc31e365cc403180132fac3ac'


def fixtures():
    refs, manifest = [], []
    for speaker in (1, 2):
        parts = []
        for sentence in range(1, 7):
            path = ROOT / 'work/samples' / f'spk{speaker}_snt{sentence}.wav'
            parts.extend([load_audio(path), np.zeros(2400, np.float32)])
            manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        wave = np.concatenate(parts)
        refs.append(wave * (.08 / np.sqrt(np.mean(wave ** 2))))
    a, b = [np.pad(w, (0, max(map(len, refs)) - len(w))) for w in refs]
    silence = np.zeros(64000, np.float32)
    arrival = [np.concatenate([a[:64000], silence, a[64000:]]), np.concatenate([silence, b])]
    rng = np.random.default_rng(20261015)
    cases = [('single', a, [a], 1.5), ('overlap', a+b, [a, b], 1.5),
             ('arrival', sum(arrival), arrival, 1.5),
             ('noisy', (a+b)*.3 + rng.normal(0, .001, len(a)).astype(np.float32), [a*.3, b*.3], 1.5),
             ('file', a+b, [a, b], 4)]
    # LJSpeech is a further public speaker, not used to choose the thresholds.
    path = ROOT / 'work/samples/LJ050-0131.wav'
    if not path.exists():
        endpoint = f'repos/speechbrain/speechbrain/contents/tests/samples/TTS/{path.name}?ref={PUBLIC_COMMIT}'
        metadata = json.loads(subprocess.check_output(['gh', 'api', endpoint], timeout=60))
        content = base64.b64decode(metadata['content'])
        if content[:4] != b'RIFF' or len(content) != metadata['size']:
            raise ValueError('Incomplete public test WAV')
        path.write_bytes(content)
    manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    c = load_audio(path)
    c *= .08 / np.sqrt(np.mean(c ** 2))
    length = max(len(a), len(c))
    aa, cc = [np.pad(w, (0, length-len(w))) for w in (a, c)]
    gap = np.zeros(24000, np.float32)
    # Three sequential identities, then a return by the first. At most two
    # simultaneous speakers, matching the separator's actual capabilities.
    schedule = [(0, a), (1, b), (2, c), (0, a)]
    timeline = [np.concatenate([w if who == target else np.zeros_like(w)
                               for who, part in schedule for w in (part, gap)])
                for target in range(3)]
    cases.extend([('heldout_single_lj', c, [c], 1.5),
                  ('heldout_pair_lj', aa+cc, [aa, cc], 1.5),
                  ('heldout_three_turns', sum(timeline), timeline, 1.5)])
    return cases, manifest


def extract(backend, wave, truths, window, context_seconds=0):
    frames = []
    processor = Processor(backend, context_seconds=context_seconds)
    size = int(window * 16000)
    for start in range(0, len(wave), size):
        tracks, vectors, durations, _ = processor.extract_features(wave[start:start+size])
        labels = []
        for track in tracks:
            qualities = []
            for source in truths:
                reference = source[start:start+len(track)]
                qualities.append(float(np.dot(track, reference) ** 2 /
                    (np.dot(track, track) * np.dot(reference, reference) + 1e-12)))
            order = np.argsort(qualities)[::-1]
            unambiguous = (qualities[order[0]] >= .25 and
                          (len(qualities) == 1 or qualities[order[0]]-qualities[order[1]] >= .1))
            labels.append(int(order[0]) if unambiguous else -1)
        frames.append({'vectors': vectors, 'durations': durations, 'labels': labels,
                       'reference_eligible': processor.reference_eligible,
                       'embedding_durations': processor.embedding_durations})
    return frames


def metrics(frames, predictions, aliases):
    pairs = [(label, aliases.get(who, who)) for frame, row in zip(frames, predictions)
             for label, who in zip(frame['labels'], row) if label >= 0]
    keys = sorted({p for _, p in pairs if p is not None})
    truth = sorted({t for t, _ in pairs})
    matrix = np.array([[sum(t == a and p == b for t, p in pairs) for b in keys] for a in truth])
    correct = int(matrix[linear_sum_assignment(-matrix)].sum()) if matrix.size else 0
    assigned = [(t, p) for t, p in pairs if p is not None]
    splits = merges = same_pairs = different_pairs = 0
    for i, (t, p) in enumerate(assigned):
        for other_t, other_p in assigned[:i]:
            same_pairs += t == other_t
            different_pairs += t != other_t
            splits += t == other_t and p != other_p
            merges += t != other_t and p == other_p
    return dict(scored_tracks=len(pairs), correctly_assigned_tracks=correct,
                unassigned_tracks=sum(p is None for _, p in pairs),
                excluded_ambiguous_tracks=sum(label < 0 for f in frames for label in f['labels']),
                false_split_pairs=int(splits), same_speaker_pairs=int(same_pairs),
                false_merge_pairs=int(merges), different_speaker_pairs=int(different_pairs))


def evaluate(cls, frames, reverse=False, repetitions=1, legacy=False):
    registry, predictions = cls(), []
    repeated = frames * repetitions
    for frame in repeated:
        order = list(range(len(frame['vectors'])))
        if reverse:
            order.reverse()
        parameters = inspect.signature(registry.assign).parameters
        kwargs = {}
        if 'reference_eligible' in parameters:
            kwargs['reference_eligible'] = frame.get('reference_eligible', True)
        if 'embedding_durations' in parameters:
            kwargs['embedding_durations'] = [frame.get('embedding_durations', frame['durations'])[i] for i in order]
        result = registry.assign([frame['vectors'][i] for i in order], [frame['durations'][i] for i in order], **kwargs)
        row = [None] * len(order)
        for i, identity in zip(order, result):
            row[i] = identity
        predictions.append(row)
    return dict(people=len(registry.speakers),
                first_identity_block=next((i for i, row in enumerate(predictions) if any(row)), None),
                **metrics(repeated, predictions, registry.aliases),
                **({'assignments': predictions} if repetitions == 1 else {'repetitions': repetitions}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='edbf656')
    parser.add_argument('--output', default='docs/evidence/balanced-classification.json')
    args = parser.parse_args()
    cases, manifest = fixtures()
    backend, old = ModelBackend(), baseline_registry(args.baseline_ref)
    from silero_vad import load_silero_vad
    current_vad, historical_vad = backend.vad, load_silero_vad()
    results = {}
    for name, waveform, sources, window in cases:
        backend.vad = historical_vad
        original = extract(backend, waveform, sources, window, context_seconds=0)
        backend.vad = current_vad
        frames = extract(backend, waveform, sources, window, context_seconds=3)
        results[name] = dict(expected_people=len(sources), window_seconds=window,
            seconds=len(waveform)/16000, sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),
            before=evaluate(old, original, legacy=True),
            classification_only=evaluate(SpeakerRegistry, original), after=evaluate(SpeakerRegistry, frames),
            reversed_outputs=evaluate(SpeakerRegistry, frames, reverse=True),
            repeated_10_times=evaluate(SpeakerRegistry, frames, repetitions=10))
        print(name, {k: results[name][k]['people'] for k in ('before', 'after', 'reversed_outputs', 'repeated_10_times')}, flush=True)
    output = dict(baseline_ref=args.baseline_ref, public_source_commit=PUBLIC_COMMIT,
        labels='Evaluation-only maximum squared waveform correlation >= .25 and lead >= .10; ambiguous outputs excluded from classification scoring, but included in profile counts.',
        limitations='Small synthetic English fixtures. First five informed clustering thresholds; LJ failures motivated past-only context, so LJ is a development regression, not a final blind accuracy test. Before uses independent windows; after uses past-only context, so eligible/scored track counts differ. Repeated trials reuse identical features, not independent speech. No real microphone/Chinese/room-acoustic accuracy claim.',
        sources=manifest, cases=results)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    assert all(case[mode]['people'] == case['expected_people'] and case[mode]['false_merge_pairs'] == 0
               for case in results.values() for mode in ('after', 'reversed_outputs', 'repeated_10_times')), \
        'Identity regression; inspect the exported evidence, including unassigned and excluded tracks.'


if __name__ == '__main__':
    main()
