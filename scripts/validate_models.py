"""Offline public-sample evaluation. The inference API receives ONLY mixture.wav.

No microphone or playback is opened. Clean sources are used only for scoring.
This fixture is a synthetic mix of distinct utterances, not an acoustic recording.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from itertools import permutations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import soundfile as sf
from selective_hearing.audio import load_audio, process_file, export_result
from selective_hearing.models import ModelBackend


def si_sdr(reference, estimate):
    reference = reference.astype(np.float64) - np.mean(reference)
    estimate = estimate.astype(np.float64) - np.mean(estimate)
    target = reference * (np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12))
    return float(10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum((estimate - target) ** 2) + 1e-12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="work/samples")
    parser.add_argument("--output", default="results/public-validation")
    parser.add_argument("--layout", choices=["overlap", "arrival"], default="overlap")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    refs, manifest = [], []
    for speaker in (1, 2):
        parts = []
        for sentence in range(1, 7):
            path = Path(args.samples) / f"spk{speaker}_snt{sentence}.wav"
            part = load_audio(path)
            parts.extend([part, np.zeros(2400, np.float32)])
            manifest.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        signal = np.concatenate(parts)
        refs.append(signal * (0.08 / max(float(np.sqrt(np.mean(signal ** 2))), 1e-8)))
    if args.layout == "arrival":
        # A alone for 4 s, then B alone for 4 s, then both on remaining utterances.
        prefix = 4 * 16000
        a, b = refs
        refs = [np.concatenate([a[:prefix], np.zeros(prefix, np.float32), a[prefix:]]),
                np.concatenate([np.zeros(prefix, np.float32), b])]
    length = max(map(len, refs))
    refs = [np.pad(x, (0, length - len(x))) for x in refs]
    mixture = refs[0] + refs[1]
    scale = min(1, .8 / max(float(np.max(np.abs(mixture))), 1e-8))
    refs = [x * scale for x in refs]
    sf.write(folder / "input_mixture.wav", sum(refs), 16000, subtype="FLOAT")
    begin = time.perf_counter()
    backend = ModelBackend()
    load_seconds = time.perf_counter() - begin
    packet, report = process_file(folder / "input_mixture.wav", backend,
        lambda e: print(f"{e['fraction']:.0%} RTF={e['rtf']:.3f} active={e['active']} people={e['speakers']}", flush=True))
    export_result(packet, report, folder)
    identities = list(packet.tracks)
    evaluation = {"source": "SpeechBrain tests/samples/ASR, commit 9ef9a3478a9fbc9cc31e365cc403180132fac3ac",
        "fixture": f"{args.layout}: two concatenated speakers, digitally mixed; not microphone capture",
        "model_load_seconds": load_seconds, "manifest": manifest,
        "duration_seconds": report["duration_seconds"], "compute_seconds": report["compute_seconds"],
        "rtf": report["compute_seconds"] / report["duration_seconds"],
        "confirmed_speakers": len(identities), "speakers": report["speakers"],
        "peak_by_identity": {k: float(np.max(np.abs(v))) for k, v in packet.tracks.items()},
        "clipping_fraction_by_identity": {k: float(np.mean(np.abs(v) > .95)) for k, v in packet.tracks.items()}}
    if len(identities) == 2:
        # One global permutation for the whole file: chunk-wise identity swaps are penalized.
        best = max(permutations(identities), key=lambda order: sum(si_sdr(r, packet.tracks[k]) for r, k in zip(refs, order)))
        evaluation["scores"] = [{"reference": i + 1, "identity": key,
            "mixture_si_sdr_db": si_sdr(ref, packet.original),
            "output_si_sdr_db": si_sdr(ref, packet.tracks[key]),
            "improvement_db": si_sdr(ref, packet.tracks[key]) - si_sdr(ref, packet.original)}
            for i, (ref, key) in enumerate(zip(refs, best))]
    (folder / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in evaluation.items() if k != "manifest"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
