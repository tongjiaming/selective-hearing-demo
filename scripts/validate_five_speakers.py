"""Five known public speakers, different utterances on return; no microphone.

Downloads pinned public FLACs through gh, keeps audio/features in ignored work/,
and exports metrics and anonymous assignments only. Ground truth is evaluation
only and is never passed to the speaker registry.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from validate_identity_classification import ROOT, extract, evaluate, ModelBackend, SpeakerRegistry
from validate_identity_stability import baseline_registry
from selective_hearing.audio import load_audio

COMMIT = '15d828edebe06bc72b9cabc8ef8ca5ab2cb457ce'
GROUPS = {'development': ('1688', '1998', '2033', '2414', '2609'),
          'heldout': ('3005', '3080', '3331', '367', '533')}


def load_group(group):
    folder = ROOT / 'work' / ('five-speakers' if group == 'development' else 'five-holdout')
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    if not manifest_path.exists():
        manifest = []
        for who in GROUPS[group]:
            base = f'audio_data/librispeech_test-other/{who}'
            entries = json.loads(subprocess.check_output(['gh', 'api',
                f'repos/resemble-ai/Resemblyzer/contents/{base}?ref={COMMIT}'], timeout=60))
            for entry in sorted(entries, key=lambda x: x['name'])[:2]:
                data = json.loads(subprocess.check_output(['gh', 'api',
                    f"repos/resemble-ai/Resemblyzer/contents/{entry['path']}?ref={COMMIT}"], timeout=60))
                raw = base64.b64decode(data['content'])
                if len(raw) != data['size'] or raw[:4] != b'fLaC':
                    raise ValueError('Incomplete public FLAC')
                (folder / entry['name']).write_bytes(raw)
                manifest.append(dict(speaker=who, file=entry['name'], source=entry['path'],
                    commit=COMMIT, sha256=hashlib.sha256(raw).hexdigest()))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    voices = {}
    for item in manifest:
        path = folder / item['file']
        assert item['commit'] == COMMIT and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        wave = load_audio(path)
        wave *= .08 / max(float(np.sqrt(np.mean(wave ** 2))), 1e-6)
        voices.setdefault(item['speaker'], []).append(wave)
    assert set(voices) == set(GROUPS[group]) and all(len(v) == 2 for v in voices.values())
    return voices, manifest


def timeline(voices, turn_seconds=6):
    schedule = []
    keys = list(voices)
    for turn in (0, 1):
        for who in (range(len(keys)) if turn == 0 else reversed(range(len(keys)))):
            schedule.extend([(who, voices[keys[who]][turn][:int(16000 * turn_seconds)]),
                             (None, np.zeros(3200, np.float32))])
    return [np.concatenate([wave if who == target else np.zeros_like(wave)
                            for who, wave in schedule]) for target in range(len(keys))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='b4ad4de')
    parser.add_argument('--output', default='docs/evidence/five-speaker-classification.json')
    args = parser.parse_args()
    backend, old = ModelBackend(), baseline_registry(args.baseline_ref)
    results, manifests = {}, {}
    for group in GROUPS:
        voices, manifests[group] = load_group(group)
        for seconds in (6, 2.4):
            name = f'{group}_{seconds:g}s_turns'
            refs = timeline(voices, seconds)
            frames = extract(backend, sum(refs), refs, 1.5, context_seconds=3)
            cache = ROOT / 'work' / f'{name}-features.json'
            serializable = [{**f, 'vectors': [v.tolist() for v in f['vectors']]} for f in frames]
            cache.write_text(json.dumps(serializable), encoding='utf-8')
            result = dict(expected_people=5, seconds=len(refs[0])/16000,
                sha256=hashlib.sha256(sum(refs).tobytes()).hexdigest(),
                before=evaluate(old, frames), after=evaluate(SpeakerRegistry, frames),
                reversed_outputs=evaluate(SpeakerRegistry, frames, reverse=True),
                repeated_10_times=evaluate(SpeakerRegistry, frames, repetitions=10))
            results[name] = result
            print(name, {k: {a: b for a, b in result[k].items() if a != 'assignments'}
                         for k in ('before', 'after', 'repeated_10_times')}, flush=True)
    payload = dict(baseline_ref=args.baseline_ref, window_seconds=1.5, past_context_seconds=3,
        source='https://github.com/resemble-ai/Resemblyzer/tree/' + COMMIT + '/audio_data/librispeech_test-other',
        sources=manifests, cases=results,
        labels='Evaluation-only squared waveform correlation >= .25 and lead >= .10. Ambiguous tracks excluded from identity scores but included in profile counts.',
        limitations='Small synthetic English read-speech tests, not live Chinese room speech. Development 6s case informed thresholds. The second group was initially held out, then its failure informed context-based seed selection: final results are development regressions, NOT blind accuracy. Rapid turns retain failures. Identical features replayed 10 times test drift, not new independent speech. Unknown tracks and pairwise errors must be read alongside profile counts. Both versions receive the same model features.')
    (ROOT / args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
