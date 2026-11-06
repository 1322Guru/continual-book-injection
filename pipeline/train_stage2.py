"""Stage-2 trainer: continue training from an EXISTING stage-1 LoRA adapter on a new row mix,
instead of a fresh LoRA. Same base (bf16, model parallel GPU 0 + GPU 1), same packing, loss masking, system
prompt, seq 512, cosine over the stage-2 steps, warmup 10, adamw_8bit, same seed. The optimizer starts
fresh (standard for a second stage); the LoRA weights start from the given adapter.

  python -I pipeline/train_stage2.py --adapter <dir> [--epochs E] [--lr X] [--micro M] [--max-steps N] <out_dir> <rows.jsonl>...
"""
import hashlib
import json
import math
import os
import signal
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import units as U  # noqa: E402
from pipeline import ctx  # noqa: E402
from pipeline.pack import pack  # noqa: E402

signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv):
    opt = {"--epochs": 1.0, "--lr": 5e-5, "--micro": 4, "--max-steps": 0, "--adapter": ""}
    while argv and argv[0] in opt:
        opt[argv[0]] = type(opt[argv[0]])(argv[1])
        argv = argv[2:]
    if not opt["--adapter"]:
        raise SystemExit("--adapter <dir with adapter_model.safetensors> is required")
    out, srcs = argv[0], argv[1:]
    cfg, _ = U.load_config()
    T = dict(cfg["training"])
    spf = os.path.join(ROOT, T["system_prompt_file"])
    system = open(spf, encoding="utf-8").read()
    rows, per_src = [], {}
    for s in srcs:
        xs = [json.loads(l) for l in open(s, encoding="utf-8")]
        rows += xs
        per_src[s] = {"rows": len(xs), "sha256": sha(s)}
    os.makedirs(out, exist_ok=True)
    micro = opt["--micro"]
    accum = max(1, T["batch"] // micro)
    init_sha = sha(os.path.join(opt["--adapter"], "adapter_model.safetensors"))
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": 2, "init_adapter": opt["--adapter"], "init_adapter_sha256": init_sha,
            "sources": per_src, "base": T["base_full"], "dtype": "bf16", "lr": opt["--lr"], "epochs": opt["--epochs"],
            "schedule": "cosine", "warmup_steps": T["warmup_steps"], "optim": T["optim"], "micro_batch": micro, "grad_accum": accum,
            "effective_batch": micro * accum, "seq_len": T["seq_len"], "gradient_checkpointing": True,
            "device_map": "GPU 0 + GPU 1 (model parallel)", "system_prompt_sha256": sha(spf), "test_knobs": ctx.test_knobs() or None}
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, default_data_collator
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(T["base_full"])
    one_gpu = torch.cuda.device_count() == 1   # rented A6000: whole model on one card, no checkpointing (36.8 GB measured)
    if one_gpu:
        model = AutoModelForCausalLM.from_pretrained(T["base_full"], torch_dtype=torch.bfloat16, device_map={"": 0},
                                                     attn_implementation="sdpa")
    else:   # home: split over GPU 0 + GPU 1; GPU 1 keeps room for the fp32 logits
        model = AutoModelForCausalLM.from_pretrained(T["base_full"], torch_dtype=torch.bfloat16, device_map="auto",
                                                     max_memory={0: "12GiB", 1: "9GiB"}, attn_implementation="sdpa")
    meta["load_s"] = round(time.time() - t0, 1)
    meta["libs"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": __import__("transformers").__version__,
                    "peft": __import__("peft").__version__,
                    "trl": (__import__("importlib.metadata").metadata.version("trl") if __import__("importlib.util").util.find_spec("trl") else None)}
    dmap = getattr(model, "hf_device_map", None) or {"": 0}   # transformers 5.17 sets no hf_device_map for device_map={"": 0}
    meta["device_split"] = {str(k): sum(1 for v in dmap.values() if v == k) for k in set(dmap.values())}
    meta["gradient_checkpointing"] = not one_gpu
    meta["device_map"] = "single GPU" if one_gpu else "GPU 0 + GPU 1 (model parallel)"
    model.config.use_cache = False
    if not one_gpu:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, opt["--adapter"], is_trainable=True)
    meta["lora"] = {k: getattr(model.peft_config["default"], k, None) for k in ("r", "lora_alpha", "lora_dropout")}
    blocks, stats = pack(tok, rows, system, T["seq_len"], T["seed"])
    meta["pack"] = stats
    ds = Dataset.from_dict(blocks)
    steps_per_epoch = math.ceil(len(ds) / (micro * accum))
    steps = opt["--max-steps"] or math.ceil(steps_per_epoch * opt["--epochs"])
    cap = ctx.max_steps()   # test knob (smoke runs): fewer steps, adapter still saved
    if cap and not opt["--max-steps"] and cap < steps:
        steps = cap
    loss_p = os.path.join(out, "loss.jsonl")
    open(loss_p, "w").close()

    class LossLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                with open(loss_p, "a") as f:
                    f.write(json.dumps({"step": state.global_step, "of": steps, "loss": logs["loss"], "lr": logs.get("learning_rate"),
                                        "grad_norm": logs.get("grad_norm"), "elapsed_s": round(time.time() - t1, 1)}) + "\n")

    args = TrainingArguments(output_dir=os.path.join(out, "trainer"), per_device_train_batch_size=micro, gradient_accumulation_steps=accum,
                             num_train_epochs=opt["--epochs"], max_steps=opt["--max-steps"] or (steps if cap and steps == cap else -1), learning_rate=opt["--lr"],
                             lr_scheduler_type="cosine", warmup_steps=T["warmup_steps"], optim=T["optim"], bf16=True,
                             logging_steps=1 if opt["--max-steps"] else 5, save_strategy="no", report_to="none", seed=T["seed"],
                             remove_unused_columns=False, gradient_checkpointing=False, dataloader_pin_memory=False)
    t1 = time.time()
    res = Trainer(model=model, args=args, train_dataset=ds, data_collator=default_data_collator, callbacks=[LossLog()]).train()
    wall = time.time() - t1
    meta.update({"steps": steps, "steps_per_epoch": steps_per_epoch, "train_s": round(wall, 1), "s_per_step": round(wall / max(1, steps), 2),
                 "mean_loss": res.training_loss,
                 "peak_vram_gb": {str(i): round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())}})
    if opt["--max-steps"]:
        json.dump(meta, open(os.path.join(out, "bench.json"), "w"), indent=2)
        print(json.dumps({k: meta[k] for k in ("steps", "s_per_step", "peak_vram_gb", "load_s", "device_split")}))
        return
    model.save_pretrained(os.path.join(out, "adapter"))
    meta["adapter_sha256"] = sha(os.path.join(out, "adapter", "adapter_model.safetensors"))
    meta["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(meta, open(os.path.join(out, "train_meta.json"), "w"), indent=2)
    print(json.dumps({k: meta[k] for k in ("steps", "train_s", "s_per_step", "mean_loss", "peak_vram_gb")}))


if __name__ == "__main__":
    main(sys.argv[1:])
