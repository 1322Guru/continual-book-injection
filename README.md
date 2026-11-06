# continual-book-injection

## Re-runnable demo (public-domain books)

This demo trains Qwen2.5-7B-Instruct with LoRA on two public-domain books in sequence and measures how much of the first book survives training on the second, with and without replay. Everything it needs is in this repo or fetched by the one command below.

```
./run_demo.sh
```

The script builds a virtual environment from `requirements.lock`, checks every data file against `data/SHA256SUMS`, downloads the pinned base model (about 15 GB) and runs all stages. `./run_demo.sh --plan` prints exact step counts first, and `./run_demo.sh --smoke` runs every stage briefly (20 steps per stage, 10 questions per book, seed 42). It needs Python 3.12 and NVIDIA driver 580 or later. On Ubuntu 24.04 that means the apt packages `python3.12-venv`, `python3.12-dev`, `git` and `build-essential`: the first builds the virtual environment, and Triton compiles a C extension against `Python.h` during the run. The output folder was 24 GB after the run, and one merged model (about 15 GB more) exists at a time while an arm is scored.

### What it does

- **T0:** score the base model on books A and B.
- **T1:** train book A on the base model and merge. This is book A's start score.
- **Arms:** each trains book B on top of T1.
  - (a) No replay.
  - (b) Share replay: the contest's rule, replay = round(0.30 / 0.70 x book B rows, everyday excluded) = 2,325 book-A rows.
  - (c) Fixed replay: 4,700 book-A rows, which is 75% of book A's 6,289 knowledge and verbatim rows.

Replay rows are stratified by type and spread evenly over units. All training uses LoRA r 128, alpha 128, LR 3e-4, 2 epochs and batch 32. These are the stage-1 settings from the contest recipe, with no stage 2 and no trap rows. Seeds: 42 for T1 and all arms, plus 43 for arms (b) and (c).

Book A is Tsiolkovsky, "Grezy o Zemle i nebe" (126 units, 6,860 rows). Book B is Darwin in Rachinsky's 1864 translation (109 units, 5,895 rows). Each unit is one sentence. See `data/LICENSE_NOTE.md`.

### Results

Each book has 100 eval questions. Every score is shown twice: on the full set, and on the cleaned set with underspecified questions removed (89 for A, 91 for B). Ratio is the book-A score divided by the T1 score on the same set.

| Stage | A full | A ratio | B full | A cleaned | A ratio cleaned | B cleaned |
|---|---|---|---|---|---|---|
| T0 base | 6/100 | | 5/100 | 6/89 | | 5/91 |
| T1 book A | 48/100 | 1.00 | 3/100 | 42/89 | 1.00 | 3/91 |
| (a) no replay, seed 42 | 13/100 | 0.27 | 55/100 | 12/89 | 0.29 | 51/91 |
| (b) share, seed 42 | 44/100 | 0.92 | 62/100 | 40/89 | 0.95 | 57/91 |
| (b) share, seed 43 | 49/100 | 1.02 | 53/100 | 44/89 | 1.05 | 48/91 |
| (c) fixed, seed 42 | 47/100 | 0.98 | 65/100 | 42/89 | 1.00 | 59/91 |
| (c) fixed, seed 43 | 55/100 | 1.15 | 58/100 | 48/89 | 1.14 | 53/91 |

What the table shows:

- **Without replay, most of book A was lost:** 48 fell to 13, in the one seed run.
- **Both replay arms kept book A near its T1 score,** with ratios from 0.92 to 1.15.
- **Share and fixed replay did not clearly separate.** Fixed scored higher on book A at each seed (mean 51.0 against 46.5 full, 45.0 against 42.0 cleaned), but the seed ranges overlap (44 to 49 against 47 to 55) and two seeds cannot settle it.
- **The seed-to-seed spread is large:** up to 8 points on book A and 9 on book B within one arm.
- **Replay did not reduce what was learned of book B:** fixed scored 65 and 58 and share 62 and 53, against 55 with no replay on one seed. This is two seeds on small books.
- **Why share and fixed may not separate here:** the row contrast is only 2.0 times (4,700 against 2,325), against about 21 times in the contest (45,000 against 2,154).

### Hardware and time

The full demo ran on one configuration only: two GPUs in model parallel, an RTX 5070 Ti 16 GB and an RTX 5060 Ti 16 GB, under WSL2 with driver 617.42. Peak VRAM, as allocated by PyTorch, was 10.2 GB and 14.45 GB. Models were read from and written to a separate Windows drive mounted in WSL2. As a contest measurement, not from this demo, training used 36.8 GB on one A6000 48 GB. The single-GPU code path ran only as the `--smoke` check below, not as the full demo.

| Stage | Steps | s/step | Train (load included) | Merge | Score | Total |
|---|---|---|---|---|---|---|
| T0 base | | | | | 12 min | 12 min |
| T1 book A | 94 | 17.60 | 33 min | 13 min | 5 min | 52 min |
| (a) no replay, seed 42 | 80 | 17.90 | 30 min | 14 min | 6 min | 50 min |
| (b) share, seed 42 | 106 | 17.73 | 36 min | 13 min | 6 min | 56 min |
| (b) share, seed 43 | 106 | 17.73 | 36 min | 14 min | 6 min | 56 min |
| (c) fixed, seed 42 | 132 | 17.67 | 44 min | 13 min | 6 min | 64 min |
| (c) fixed, seed 43 | 132 | 17.65 | 44 min | 13 min | 6 min | 64 min |

The whole run took 5.89 hours.

`./run_demo.sh --smoke` passed from a fresh clone of commit dd59432 on this same machine, with a new virtual environment built from `requirements.lock` and a fresh download of the base model: all 17 stages, no errors, 2.06 hours of stage time. Later commits change only this README and the file modes of `data/results/`.

`./run_demo.sh --smoke` also passed from a fresh clone of the public repo at commit cfacc73 on one rented NVIDIA RTX A6000 48 GB, driver 580.178.04, Ubuntu 24.04, 4.32 CPU cores (container quota): all 17 stages, no errors, 7.68 to 7.70 s/step with gradient checkpointing off, peak VRAM 37.48 GB as allocated by PyTorch, 24.5 minutes of stage time, 30.0 minutes with setup and the model download. This is the only run on another machine, and the full demo has not been run on one.

### Caveats

- **Small books.** Each is about a hundred single sentences sampled from a longer text, not continuous prose.
- **A re-run will not match these numbers exactly.** Fixed seeds do not make training bit-identical. Two 20-step smoke runs with the same seed and identical input rows produced different adapters, and 4 to 9 of 20 answers differed per trained stage. Answers from the untrained base model were identical (0 of 20 differed). The seed spread in the table therefore includes this run-to-run noise.
- **Few seeds.** Arms (b) and (c) ran two seeds, arm (a) one, and T1 one.
- **Every ratio is relative to a single T1 run.** All arms start from that one T1, and T1's own seed spread is not measured.
- **The arms differ in more than replay.** More replay also means more training steps.
- **A ratio above 1 does not mean there was no forgetting.** The replay arms keep training book A, so its score can rise.
- **The eval is recall of trained facts in new wording, not held-out facts.** Every unit is in the training rows. Questions sharing 8 consecutive words with that book's training rows were dropped.
- **Eval questions are model-written.** Gemma 4 31B wrote them, a different family from the Qwen teacher. Some are vague out of context, and the cleaned set removes only what one fixed rule catches.
- **Matcher scoring.** The scorer is a deterministic key-fact match with no LLM judge. It can miss correct paraphrases and can pass a long answer that contains the key words. It also counts a range answer containing the reference as a match. This applies to all arms equally.
- **Scores are not comparable with the contest report below.** The demo uses Transformers in bf16 with greedy decoding, the contest used a q8 GGUF stack, and the books, questions and scorer all differ.
- **Everyday answers were generated under a different system prompt** from the one in `data/system_prompt.txt`.
- **The share arm re-samples stored rows.** The contest share run regenerated part of its replay.
- **Not measured here:** made-up answers, refusals and general ability. The demo has no trap rows.

The scorer changes and the underspecified-question rule were committed before the full run started (commit e50b18a). The rule is in `demo/flag_eval.py`, and the flagged list is `data/eval/underspecified.jsonl`. All answers and raw results are in `data/results/`.

## Contest report

A report, with code excerpts, on step-wise LoRA training of Qwen2.5-7B-Instruct on three books in sequence, with forgetting of the first book measured after the third.

**The contest pipeline below is not runnable as published. The demo above is.** The report below is a write-up of measured contest results plus the modules that implement the replay and training steps. The contest's data, driver script, row generation and scorer are not here.

Author: Gursimran Singh. License: Apache-2.0.

## The task

The work was done for a closed-book knowledge-injection contest:

- Base model: Qwen2.5-7B-Instruct, trained with LoRA.
- Three books are trained in sequence. The knowledge has to live in the weights: closed book, no retrieval at answer time.
- After book 3, the model must still answer questions about book 1.
- The model must also reject questions built on a false detail, and must not start refusing true questions or everyday requests.

## The method

Each book trains a fresh LoRA on the previous book's merged weights, in two stages. Stage 1 runs 2 epochs at LR 3e-4 with r 128, alpha 128. Stage 2 continues the same adapter for 1 epoch at LR 5e-5. The adapter is then merged into the bf16 weights, and the next book starts from there.

**Fixed-count replay, re-sampled from stored rows.** Books 2 and 3 take a fixed number of book-1 rows, whatever the size of the new book: 45,000 in stage 1 and 3,600 in stage 2 in the final recipe. Other earlier books contribute 40% of their rows in stage 1 and 50% in stage 2. Rows are sampled with a fixed seed from what is already stored, so replay needs no generation calls. Knowledge rows are spread one per unit per round, so the sample covers the whole book, and trap rows stay together with their paired twin rows.

**Presupposition-frame trap rows.** Most of the false-premise failures came from one frame: a WH-question that takes the false detail as settled and asks about something downstream of it. The training rows use that frame in pairs. The trap question carries a false detail and its answer rejects the premise. The twin question carries the true detail and its answer responds normally, so the model cannot learn "always reject". The code that generates these rows is withheld (see below).

## Results

All three runs are scored on book 1 after book 3 has been trained. Book-1 score is out of 150 and the retention set is out of 100. Ratio is the book-1 score after book 3 divided by that run's own start score.

| Run | Book-1 start score | Book-1 score after book 3 | Ratio | Retention set before -> after | Made-up (of 60 traps) | Twin refusals |
|---|---|---|---|---|---|---|
| Share replay | 64 | 47 | 0.73 | 52 -> 42 | 36 | 0 |
| 30k fixed replay | 71 | 66 | 0.93 | 47 -> 41 | 18 | 1 |
| 45k fixed replay | 71 | 78 | 1.10 | 47 -> 48 | 22 | 1 |

How to read this table:

- **Measured on our own probe harness.** The questions were written by Claude for this work. Client data and the contest's exam are not included. The client's own run was not performed, so these are not contest scores.
- **One run each.** Run-to-run variance is unmeasured. The only data point is that two book-1 models from the same recipe scored 64 and 71, a 7-point gap. That is more than half of the 12-point gap between the 30k and 45k runs, so the order of those two runs is not established.
- **Two different start models.** The share run started from the 64 model, trained end to end on a rented A6000. The fixed-replay runs started from the 71 model, trained at home. This is not a clean one-variable comparison. On a common denominator of 71, the three ratios are 0.66, 0.93 and 1.10.
- **The share run also differs in its replay source.** Its replay rows were partly regenerated by the writer model rather than only re-sampled.
- **A ratio above 1 does not mean there was no forgetting.** Later books keep training book 1 through replay, so the book-1 score can rise. In the 45k run the book-1 score rose 7 while the retention set rose 1.
- **Absolute scores depend on the scorer.** The rule-based matcher is strict. On a separate 150-question set for the 45k model, it counted 73 correct and a Claude Opus judge counted 101, with 80% agreement.

**Made-up answers:** the untrained base model made up an answer on 18 of 60 traps. With fixed replay, our models made up 18 to 24 of 60 (36 under share replay), and the contest limit was 2. Knowledge training alone raised the count to 49. Trap training brought it back to about the base level, and no model came close to the limit.

**Setup:** books 2 and 3 were public-domain mock books, not contest books: Tsiolkovsky, and Darwin in Rachinsky's translation, as slices of 145 and 119 units. Own stage-1 rows were 8,614 and 6,947, against about 74,500 for book 1. All material was in Russian. Answers were scored at temperature 0 from a q8_0 GGUF of the merged model.

Generated training rows were written by an open-weight Qwen teacher (Qwen3.8-27B), and trap-type rows were filtered by Gemma. Everyday prompts came from the same teacher, everyday answers came from the base model itself, and verbatim rows use the book text. No generated row text came from a closed model. Claude wrote the evaluation probes and their reference answers and served as one judge. During development it also reviewed samples of trap rows, which led to generator changes and the removal of 3 pairs from book 1's rows.

## What failed

- **Share-based replay.** Setting replay as a share of the new book (30%, then 40%) gave too few book-1 rows, because the later books are small. Book 3 received only 2,154 book-1 rows, and the ratio fell to 0.73. Fixed counts replaced it.
- **Presupposition rows plateau.** More of these rows moved the made-up count at first, then stopped helping: it stayed at 18 to 24 of 60, and 22 of the 45 WH-frame traps was the best result on that subset. Higher doses cost accuracy on the true twin questions. Restraint is the unsolved part of this work.

## What is included

| File | Why it is here |
|---|---|
| `pipeline/mix.py` | Replay and row mixing: fixed counts, the seeded re-sample, stage 1 and stage 2 row sets |
| `pipeline/governor.py` | Projects a book's GPU hours, counting replay, and lowers rows per unit to fit a cap |
| `pipeline/train_bf16.py` | Stage 1: LoRA on bf16 weights |
| `pipeline/train_stage2.py` | Stage 2: continues the stage-1 adapter on a new row mix |
| `pipeline/pack.py` | Row packing and loss masking, used by both stages |
| `pipeline/merge_export.py` | Merges the adapter into the weights, shard by shard, and converts to q8_0 GGUF |
| `pipeline/ctx.py`, `pipeline/units.py` | Paths and the config loader that the modules above import |
| `example_config.yaml` | Replay and training keys only |

The modules are edited excerpts, not the code as it ran. Internal names were changed, and `governor.py` had its API cost planning removed. `governor.py` also reads `governor:` and `book:` config blocks that `example_config.yaml` does not include. The demo above ran these modules in their edited form: `pipeline/train_bf16.py`, `pipeline/pack.py`, `pipeline/ctx.py`, `pipeline/units.py`, the merge step of `pipeline/merge_export.py`, and the `alloc` and `knowledge_per_unit` functions of `pipeline/mix.py`. These have still not been run in this form: `pipeline/governor.py`, `pipeline/train_stage2.py`, the GGUF step of `pipeline/merge_export.py`, and the rest of `pipeline/mix.py`.

## What is withheld

The following embed or depend on client material, so they are not published: the trap and twin row generators, the writer prompts, the knowledge row generator, the scorer, the probe and exam question files, the system prompt, the book files, all training rows, and all call logs. The full run config is also withheld. No model weights are published.

## License

Copyright 2026 Gursimran Singh. Licensed under the Apache License, Version 2.0. See `LICENSE`.
