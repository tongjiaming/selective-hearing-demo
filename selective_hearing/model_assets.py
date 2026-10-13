"""Only model weights are downloaded; no remote Python/YAML is executed."""
ASSETS = {
    "conv_tasnet.pt": {
        "url": "https://download.pytorch.org/torchaudio/models/conv_tasnet_base_libri2mix.pt",
        "sha256": "ddf97d8341f616c6d8d1a4b20dd51042760392c6543e9eaac276c9f96eba0f0e",
        "size": 20049107,
    },
    "ecapa.ckpt": {
        "url": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt",
        "sha256": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
        "size": 83316686,
    },
}
