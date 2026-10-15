"""Held-out public speech comparison of frozen vs persistent adaptive profiles.

Enrollment: first 1.5 s of sentence 1. Learning: non-overlapping live-size
windows from sentences 2 and 3. Test: sentences 4, 5, 6, never used for learning.
This isolates identity learning, not mixed-speaker separation or microphone quality.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selective_hearing.audio import load_audio
from selective_hearing.core import SpeakerRegistry, unit
from selective_hearing.models import ModelBackend
from selective_hearing.profiles import ProfileStore


def main():
    backend = ModelBackend()
    manifest, clips = [], {}
    for person in (1, 2):
        for sentence in range(1, 7):
            path = ROOT / 'work/samples' / f'spk{person}_snt{sentence}.wav'
            manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            waveform = load_audio(path)
            features = []
            for offset in range(0, len(waveform), 24000):
                part = waveform[offset:offset + 24000]
                if len(part) < 16000:
                    continue
                speech = backend.speech(part)
                if len(speech) < 12000:
                    continue
                features.append((offset / 16000, unit(backend.embedding(speech)), len(speech) / 16000))
            clips[person, sentence] = features
    with tempfile.TemporaryDirectory(prefix='hearing-profile-validation-') as temp:
        store = ProfileStore(Path(temp) / 'adaptive.sqlite3')
        frozen = SpeakerRegistry(learning=False)
        adaptive = SpeakerRegistry(store=store)
        for person in (1, 2):
            offset, vector, seconds = clips[person, 1][0]
            assert offset == 0
            frozen._register(vector, seconds)
            identity = adaptive._register(vector, seconds)
            store.rename(identity, f'Test person {person}')
        for sentence in (2, 3):
            for person in (1, 2):
                for _, vector, seconds in clips[person, sentence]:
                    frozen.assign([vector], [seconds])
                    adaptive.assign([vector], [seconds])
        restored = SpeakerRegistry(store=ProfileStore(store.path), learning=False)
        assert [s.id for s in restored.speakers] == [s.id for s in adaptive.speakers]
        assert all(a.learned_observations == b.learned_observations for a, b in zip(adaptive.speakers, restored.speakers))
        rows = []
        for person in (1, 2):
            expected = f'speaker_{person}'
            for sentence in (4, 5, 6):
                for offset, vector, seconds in clips[person, sentence]:
                    row = {'person': person, 'sentence': sentence, 'offset_seconds': offset}
                    for label, source in [('frozen', frozen), ('learned', restored)]:
                        # Fresh registry per probe: test data cannot create a
                        # template or candidate later reused by another probe.
                        probe = SpeakerRegistry(learning=False)
                        probe.speakers = copy.deepcopy(source.speakers)
                        scores = {s.id: s.similarity(vector) for s in probe.speakers}
                        row[label + '_identity'] = probe.assign([vector], [seconds])[0]
                        row[label + '_target_similarity'] = round(scores[expected], 4)
                    rows.append(row)
        summary = {'test_segments': len(rows), 'restart_restored_profiles': True,
                   'names_restored': [r['name'] for r in store.summaries()],
                   'learned_observations': [s.learned_observations for s in restored.speakers]}
        for label in ('frozen', 'learned'):
            summary[label + '_correct'] = sum(r[label + '_identity'] == f"speaker_{r['person']}" for r in rows)
            summary[label + '_wrong_person'] = sum(r[label + '_identity'] not in (None, f"speaker_{r['person']}") for r in rows)
            summary[label + '_mean_target_similarity'] = round(float(np.mean([r[label + '_target_similarity'] for r in rows])), 4)
        report = {'source': 'SpeechBrain public ASR samples', 'protocol': __doc__,
                  'limitations': 'Two English speakers, eight eligible held-out segments; clean speech features only. Not proof of monotonic improvement in a real room.',
                  'manifest': manifest, 'summary': summary, 'held_out': rows}
        destination = ROOT / 'docs/evidence/profile-learning.json'
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        assert sum(summary['learned_observations']) > 0
        assert summary['learned_correct'] >= summary['frozen_correct']
        assert summary['learned_wrong_person'] == 0


if __name__ == '__main__':
    main()
