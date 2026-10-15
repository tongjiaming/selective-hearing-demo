"""Fingerprint shipped program code so debug updates start with fresh learning."""
from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def learning_revision():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'selective_hearing').glob('*.py'))
    paths += list(root.glob('requirements*.txt'))
    paths += [root / 'run.ps1', root / 'run.cmd']
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode('utf-8'))
            digest.update(b'\0')
            digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
            digest.update(b'\0')
    return digest.hexdigest()
