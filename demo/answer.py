"""Answer eval questions with Transformers: bf16, greedy, one question at a time.

  python demo/answer.py <model_dir> <out.jsonl> <n_or_0> <eval.jsonl>...
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_NEW = 128


def main(model_dir, out, n, *evals):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    system = open(os.path.join(ROOT, "data", "system_prompt.txt"), encoding="utf-8").read()
    tok = AutoTokenizer.from_pretrained(model_dir)
    if torch.cuda.device_count() == 1:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto",
                                                     max_memory={0: "12GiB", 1: "12GiB"})
    model.eval()
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in evals:
            items = [json.loads(line) for line in open(e, encoding="utf-8")]
            for r in items[:int(n) or None]:
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["question"]}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
                with torch.no_grad():
                    g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                                       repetition_penalty=1.0)
                text = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                f.write(json.dumps({"id": r["id"], "answer": text}, ensure_ascii=False) + "\n")
                f.flush()
    print(json.dumps({"answer_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main(*sys.argv[1:])
