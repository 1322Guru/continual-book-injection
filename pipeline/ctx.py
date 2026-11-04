"""Run context: every path a phase reads or writes comes from here, never from a hard-coded folder.

The driver script (not included) exports the per-book values before each phase:
  CBI_BOOK_OUT   this book's output folder (required by every in-run phase)
  CBI_BOOK_ID    the book id (unit ids are "<book id>-u001-001"); default: the folder name of CBI_BOOK_OUT
  CBI_PREV       the previous book's output folder (books 2 and 3: merged weights, replay sources, everyday rows)
  CBI_MODELS     model folder (default <repo>/models), CBI_TOOLS llama.cpp folder (default <repo>/tools)
Test-only knobs (never set by a real run; recorded in every manifest when set):
  CBI_SLICE      keep every k-th unit (0.05 -> every 20th), CBI_MAX_STEPS cap optimizer steps per stage
"""
import json
import math
import os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def models_dir():
    return os.path.abspath(env("CBI_MODELS", os.path.join(PKG, "models")))


def tools_dir():
    return os.path.abspath(env("CBI_TOOLS", os.path.join(PKG, "tools")))


def llama_cpp():
    return os.path.join(tools_dir(), "llama.cpp")


def llama_server_bin():
    return os.path.join(llama_cpp(), "build", "bin", "llama-server")


def book_out():
    v = env("CBI_BOOK_OUT")
    if not v:
        raise SystemExit("CBI_BOOK_OUT is not set: the driver script sets it")
    return os.path.abspath(v)


def book_id():
    return env("CBI_BOOK_ID") or os.path.basename(book_out().rstrip("/"))


def prev_out():
    v = env("CBI_PREV")
    return os.path.abspath(v) if v else None


def path(*parts, base=None):
    """A path inside this book's output folder (or inside `base`, another book's folder)."""
    return os.path.join(base or book_out(), *parts)


def chain(base=None):
    """This book's folder and every earlier book's, oldest first, following each folder's prev.txt link."""
    out, cur, seen = [], os.path.abspath(base or book_out()), set()
    while cur and cur not in seen:
        seen.add(cur)
        out.append(cur)
        p = os.path.join(cur, "prev.txt")
        cur = open(p).read().strip() if os.path.exists(p) else None
    return list(reversed(out))


def slice_k():
    s = env("CBI_SLICE")
    return max(1, round(1 / float(s))) if s else 1


def max_steps():
    return int(env("CBI_MAX_STEPS", "0") or 0)


def test_knobs():
    return {k: os.environ[k] for k in ("CBI_SLICE", "CBI_SLICE_MODE", "CBI_MAX_STEPS") if os.environ.get(k)}


def units_file(base=None):
    return path("units", "units.jsonl", base=base)


def load_units(base=None):
    """The book's trained units, sliced when CBI_SLICE is set."""
    us = [json.loads(l) for l in open(units_file(base), encoding="utf-8")]
    k = slice_k()
    if k > 1 and env("CBI_SLICE_MODE") == "sections":   # whole sections (keeps same-section splice pairs)
        secs = sorted({u["chapter_number"] for u in us})
        keep = set(secs[::k])
        return [u for u in us if u["chapter_number"] in keep]
    return us[::k] if k > 1 else us


def base_model():
    """Weights this book trains on and merges into: the previous book's merged model, else the pinned HF base."""
    p = prev_out()
    if p:
        m = os.path.join(p, "merged")
        if not os.path.exists(os.path.join(m, "model.safetensors.index.json")):
            raise SystemExit(f"previous book's merged model is missing: {m}")
        return m
    return os.path.join(models_dir(), "Qwen2.5-7B-Instruct")


def hf_base():
    return os.path.join(models_dir(), "Qwen2.5-7B-Instruct")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def write_jl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)


def decision(rule, inputs, action):
    """decisions.jsonl of the book: every automatic choice the run makes (governor, fallback, gate logs)."""
    import time
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "book": book_id(), "rule": rule, "inputs": inputs, "action": action}
    with open(path("decisions.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def ceil_div(a, b):
    return int(math.ceil(a / b))
