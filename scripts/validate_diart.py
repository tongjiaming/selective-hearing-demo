"""Evaluate the optional full diart pipeline on a local recording, never a mic."""
import argparse
import json
from pathlib import Path
import sys

for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selective_hearing.diart_probe import check_environment, build_pipeline, run_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check imports/login without downloading models')
    parser.add_argument('--file', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/diart-validation'))
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--step', type=float, default=.5)
    parser.add_argument('--latency', type=float, default=1.5)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
        return
    if args.file is None:
        parser.error('Specify --check or --file path/to/audio.wav')
    pipeline = build_pipeline(args.duration, args.step, args.latency)
    report = run_file(args.file, pipeline, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != 'segments'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
