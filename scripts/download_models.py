"""Download public pretrained weights with size and digest validation."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selective_hearing.model_assets import ASSETS


def main():
    import requests
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-endpoint", default="https://huggingface.co",
                        choices=["https://huggingface.co", "https://hf-mirror.com"],
                        help="镜像为可选下载源；不发送 token，仍验证同一 SHA256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "models"
    root.mkdir(exist_ok=True)
    for name, asset in ASSETS.items():
        path = root / name
        if path.exists() and path.stat().st_size == asset["size"]:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not asset["sha256"] or digest == asset["sha256"]:
                print(f"Verified {name}: {digest}", flush=True)
                continue
        url = asset["url"].replace("https://huggingface.co", args.hf_endpoint)
        partial = path.with_suffix(path.suffix + ".partial")
        print(f"Downloading {name} ({asset['size'] / 1e6:.1f} MB)", flush=True)
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            size = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if size != asset["size"] or (asset["sha256"] and digest.hexdigest() != asset["sha256"]):
            raise ValueError(f"下载完整性检查失败：{name} ({size} bytes)")
        partial.replace(path)
        print(f"Verified {name}: {digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
