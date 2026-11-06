"""Exact step counts per training stage from the packed rows, and a time estimate.

  python demo/plan.py <base_model_dir> [s_per_step]
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
import mix_demo as M  # noqa: E402
from pipeline.pack import pack  # noqa: E402

STAGES = [("t1", None, 42), ("arm_none_s42", "none", 42), ("arm_share_s42", "share", 42), ("arm_fixed_s42", "fixed", 42),
          ("arm_share_s43", "share", 43), ("arm_fixed_s43", "fixed", 43)]
BATCH, SEQ, EPOCHS = 32, 512, 2


def main(base, s_step=17.4):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    system = open(os.path.join(ROOT, "data", "system_prompt.txt"), encoding="utf-8").read()
    total = 0
    print(f"{'stage':16}{'own':>7}{'replay':>8}{'rows':>7}{'blocks':>8}{'steps':>7}{'minutes':>9}")
    for name, arm, seed in STAGES:
        rows, rep = M.build("t1") if arm is None else M.build("arm", arm, seed)
        _, st = pack(tok, rows, system, SEQ, seed)
        steps = math.ceil(math.ceil(st["blocks"] / BATCH) * EPOCHS)
        total += steps
        print(f"{name:16}{rep['own']:>7}{rep['replay']:>8}{rep['rows']:>7}{st['blocks']:>8}{steps:>7}{steps * s_step / 60:>9.1f}")
    print(json.dumps({"total_steps": total, "training_hours": round(total * s_step / 3600, 2), "s_per_step": s_step}))


if __name__ == "__main__":
    main(sys.argv[1], *(float(x) for x in sys.argv[2:3]))
