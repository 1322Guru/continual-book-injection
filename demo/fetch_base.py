"""Download the pinned base model into <models>/Qwen2.5-7B-Instruct and check the sha256 of every weight file.

  python demo/fetch_base.py <models_dir>
"""
import hashlib
import os
import sys

REPO, REV = "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"
FILES = {
    "model-00001-of-00004.safetensors": "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    "model-00002-of-00004.safetensors": "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    "model-00003-of-00004.safetensors": "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    "model-00004-of-00004.safetensors": "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    "config.json": None, "generation_config.json": None, "model.safetensors.index.json": None, "tokenizer.json": None,
    "tokenizer_config.json": None, "vocab.json": None, "merges.txt": None}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(models):
    from huggingface_hub import hf_hub_download
    d = os.path.join(os.path.abspath(models), "Qwen2.5-7B-Instruct")
    os.makedirs(d, exist_ok=True)
    for fn, want in FILES.items():
        dst = os.path.join(d, fn)
        if os.path.exists(dst) and (want is None or sha(dst) == want):
            continue
        p = hf_hub_download(repo_id=REPO, revision=REV, filename=fn, local_dir=d)
        if want and sha(p) != want:
            raise SystemExit(f"sha256 mismatch: {fn}")
    print(f"base model ready: {d}")


if __name__ == "__main__":
    main(sys.argv[1])
