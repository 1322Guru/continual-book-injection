"""Merge a LoRA adapter into this book's base weights (bf16, shard by shard on CPU) and convert to q8_0 GGUF
with the pinned llama.cpp checkout (tools/llama.cpp @ 23b0202a). The base is the pinned HF base for book 1 and the
previous book's merged model for books 2 and 3 (fresh LoRA on the merged weights). Output folders get
SOURCE.txt and a sha256 file.

  python -I pipeline/merge_export.py merge <adapter_dir> <out_dir>
  python -I pipeline/merge_export.py gguf <hf_dir> <out.gguf>        (also used for the clean base)
"""
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import ctx  # noqa: E402
from pipeline import units as U  # noqa: E402

signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
LLAMA_CPP = ctx.llama_cpp()
LLAMA_COMMIT = "23b0202a189c44a54625aadcb37a946dd1d6278d"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge(adapter, out):
    """W' = W + (alpha / r) * B @ A for every LoRA target."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    cfg, _ = U.load_config()
    base = cfg["training"]["base_full"]
    acfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    scale = acfg["lora_alpha"] / acfg["r"]
    t0 = time.time()
    with safe_open(os.path.join(adapter, "adapter_model.safetensors"), "pt") as f:
        lora = {k: f.get_tensor(k) for k in f.keys()}
    pairs = {k.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight"): (lora[k], lora[k.replace(".lora_A.", ".lora_B.")])
             for k in lora if ".lora_A." in k}
    os.makedirs(out, exist_ok=True)
    n = 0
    for shard in sorted(glob.glob(os.path.join(base, "model-*.safetensors"))):
        with safe_open(shard, "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
            meta = f.metadata()
        for k, w in tensors.items():
            if k in pairs:
                a, b = pairs[k]
                tensors[k] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
                n += 1
        save_file(tensors, os.path.join(out, os.path.basename(shard)), metadata=meta or {"format": "pt"})
        del tensors
    if n != len(pairs):
        raise SystemExit(f"merged {n} targets but the adapter has {len(pairs)} LoRA pairs")
    for fn in ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy2(os.path.join(base, fn), out)
    open(os.path.join(out, "SOURCE.txt"), "w").write(
        f"Merged model: base {base} (book chain from Qwen/Qwen2.5-7B-Instruct @ {cfg['training']['base_commit']}) + LoRA adapter {adapter}\n"
        f"adapter sha256 {sha_file(os.path.join(adapter, 'adapter_model.safetensors'))}, scale {scale}, {n} targets, bf16, merged "
        f"{time.strftime('%Y-%m-%d %H:%M')} by pipeline/merge_export.py\n")
    with open(os.path.join(out, "sha256.txt"), "w") as f:
        for fn in sorted(os.listdir(out)):
            if fn.endswith(".safetensors"):
                f.write(f"{sha_file(os.path.join(out, fn))}  {fn}\n")
    print(json.dumps({"targets_merged": n, "scale": scale, "total_s": round(time.time() - t0, 1), "out": out}))


def gguf(hf_dir, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=os.path.join(LLAMA_CPP, "gguf-py"))
    log = out_file + ".convert.log"
    p = subprocess.run([sys.executable, os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py"), hf_dir, "--outtype", "q8_0", "--outfile", out_file],
                       stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    if p.returncode:
        raise SystemExit(f"convert_hf_to_gguf failed ({p.returncode}), see {log}")
    digest = sha_file(out_file)
    open(out_file + ".sha256", "w").write(f"{digest}  {os.path.basename(out_file)}\n")
    src = os.path.join(os.path.dirname(out_file), "SOURCE.txt")
    with open(src, "a") as f:
        f.write(f"{os.path.basename(out_file)}: q8_0 GGUF of {hf_dir}, llama.cpp {LLAMA_COMMIT} convert_hf_to_gguf.py, sha256 {digest}, "
                f"{os.path.getsize(out_file)} bytes, {time.strftime('%Y-%m-%d %H:%M')}\n")
    print(json.dumps({"total_s": round(time.time() - t0, 1), "out": out_file, "gb": round(os.path.getsize(out_file) / 1e9, 2), "sha256": digest}))


if __name__ == "__main__":
    a = sys.argv
    {"merge": merge, "gguf": gguf}[a[1]](a[2], a[3])
