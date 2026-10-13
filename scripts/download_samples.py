"""Download 12 public SpeechBrain test utterances, never user recordings."""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import requests

COMMIT = "9ef9a3478a9fbc9cc31e365cc403180132fac3ac"
folder = Path(__file__).resolve().parents[1] / "work" / "samples"
folder.mkdir(parents=True, exist_ok=True)
for speaker in (1, 2):
    for sentence in range(1, 7):
        name = f"spk{speaker}_snt{sentence}.wav"
        endpoint = f"repos/speechbrain/speechbrain/contents/tests/samples/ASR/{name}?ref={COMMIT}"
        if shutil.which("gh"):
            # gh handles its own GitHub auth; no token is read or printed by this script.
            result = subprocess.run(["gh", "api", endpoint], capture_output=True, check=True, timeout=60)
            metadata = json.loads(result.stdout)
        else:
            response = requests.get(f"https://api.github.com/{endpoint}", timeout=(15, 45))
            response.raise_for_status()
            metadata = response.json()
        content = base64.b64decode(metadata["content"])
        if content[:4] != b"RIFF" or len(content) != metadata["size"]:
            raise ValueError(f"Incomplete WAV: {name}")
        (folder / name).write_bytes(content)
        print(f"Downloaded public sample {name}: {len(content)} bytes", flush=True)
