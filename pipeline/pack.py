"""Row packing for training: the model's own
chat template, every row = system prompt + question + answer, loss on the assistant tokens only, rows
shuffled with the training seed, packed into seq-length blocks.
"""
import math
import random


def pack(tok, rows, system, seq, seed):
    """Tokenise with the model's own chat template, mask everything before the assistant answer, shuffle, pack."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    ids, labels, lens = [], [], []
    for r in rows:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["question"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": r["answer"]}], tokenize=False)
        if not full.startswith(prompt):
            raise SystemExit("chat template: the prompt is not a prefix of the full conversation")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        if f_ids[:len(p_ids)] != p_ids:
            raise SystemExit("tokenisation: prompt tokens differ inside the full conversation")
        ids += f_ids
        labels += [-100] * len(p_ids) + f_ids[len(p_ids):]
        lens.append(len(f_ids))
    n_blocks = math.ceil(len(ids) / seq)
    pad = n_blocks * seq - len(ids)
    att = [1] * len(ids) + [0] * pad
    ids += [tok.pad_token_id] * pad
    labels += [-100] * pad
    blocks = {"input_ids": [ids[i * seq:(i + 1) * seq] for i in range(n_blocks)],
              "attention_mask": [att[i * seq:(i + 1) * seq] for i in range(n_blocks)],
              "labels": [labels[i * seq:(i + 1) * seq] for i in range(n_blocks)]}
    stats = {"rows": len(rows), "tokens": len(ids) - pad, "loss_tokens": sum(1 for x in labels if x != -100), "blocks": n_blocks,
             "pad_tokens": pad, "mean_row_tokens": round(sum(lens) / max(1, len(lens)), 1), "max_row_tokens": max(lens) if lens else 0}
    return blocks, stats
