"""How data/eval/ was made. Not part of run_demo.sh: the frozen files ship in the repo. Needs a llama-server with
Gemma 4 31B (google/gemma-4-31B-it-qat-q4_0-gguf @ 59dde24) on the given port.

  python demo/make_eval.py <port> <out_dir>

One call per unit, temperature 0: three question / answer candidates, the answer copied from the unit. The first
candidate of a unit that passes every check is kept, one question per unit, then 100 units per book at most (seeded).
Checks: answer found word for word in the unit, at most 6 words, has a content word of 4+ letters or a number, no key
word of the answer in the question, no source word in the question, and no 8 consecutive words shared with that
book's training rows.
"""
import collections
import json
import os
import random
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scorer  # noqa: E402

SYSTEM = """You write closed-book quiz questions in Russian about one sentence from a book.
Write 3 different question and answer pairs. Rules:
1. The question is answerable from the sentence alone and makes sense on its own to a reader who knows the book: name the subject, no pronouns such as he, this or they without a referent.
2. Never mention the book, the text, the sentence, the passage, the chapter or the author.
3. Use your own wording in the question. Do not copy long phrases from the sentence.
4. The answer is a short specific fact, 1 to 5 words, copied exactly from the sentence: a number, a name, a term or a concrete detail. The answer words must not appear in the question.
Output only a JSON array of objects with keys "question" and "answer"."""
PER_BOOK, SEED = 100, 20261105
SRC = set(scorer.LEX["source_words"])


def call(port, unit):
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Section: {unit['title']}\nSentence: {unit['text']}"}],
            "temperature": 0, "seed": 0, "max_tokens": 700, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", text, re.S)
    try:
        return [x for x in json.loads(m.group(0)) if isinstance(x, dict)] if m else []
    except json.JSONDecodeError:
        return []


def grams(w, n=8):
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def row_grams(book):
    g = set()
    for t in ("knowledge", "verbatim"):
        for line in open(os.path.join(ROOT, "data", "rows", f"book_{book}_{t}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            g |= grams(scorer.words(r["question"])) | grams(scorer.words(r["answer"]))
    return g


def reason(c, unit, g8):
    q, a = str(c.get("question") or "").strip(), str(c.get("answer") or "").strip()
    uw, aw, qw = scorer.words(unit["text"]), scorer.words(a), scorer.words(q)
    if not q.endswith("?") or not aw:
        return "format"
    if " ".join(aw) not in " ".join(uw):
        return "answer_not_in_unit"
    if len(aw) > 6:
        return "answer_too_long"
    keys = [w for w in aw if w not in scorer.STOP]
    if not any(len(w) >= 4 or w.isdigit() for w in keys):
        return "answer_not_specific"
    if any(scorer.same(k, w) for k in keys for w in qw):
        return "answer_in_question"
    if SRC & set(qw):
        return "source_word"
    if grams(qw) & g8:
        return "8gram_with_training_rows"
    return None


def main(port, out):
    os.makedirs(out, exist_ok=True)
    rep = {}
    for book in "ab":
        g8 = row_grams(book)
        units = [json.loads(line) for line in open(os.path.join(ROOT, "data", "books", f"book_{book}_units.jsonl"), encoding="utf-8")]
        kept, why = [], collections.Counter()
        for i, u in enumerate(units):
            for c in call(port, u):
                r = reason(c, u, g8)
                if r:
                    why[r] += 1
                else:
                    kept.append({"unit_id": u["id"], "question": c["question"].strip(), "answer": c["answer"].strip()})
                    break
            print(book, i + 1, len(units), len(kept), flush=True)
        kept = sorted(random.Random(SEED).sample(kept, min(PER_BOOK, len(kept))), key=lambda r: r["unit_id"])
        with open(os.path.join(out, f"book_{book}.jsonl"), "w", encoding="utf-8") as f:
            for i, r in enumerate(kept):
                f.write(json.dumps({"id": f"{book}-q{i + 1:03d}", **r}, ensure_ascii=False) + "\n")
        rep[book] = {"units": len(units), "kept": len(kept), "dropped_candidates": dict(why)}
    print(json.dumps(rep))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
