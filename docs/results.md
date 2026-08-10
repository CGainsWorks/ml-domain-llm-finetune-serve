# Results

Every number here is reproducible from a script in this repository. Where a
number does not exist yet, this file says so rather than estimating one.

---

## 1. Baseline — measured

Base model, no fine-tuning.

```bash
make baseline     # → evals/results/baseline.json
```

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798` |
| Hardware | 1 × RTX 4060 (8 GB), bf16 |
| Decoding | greedy (`temperature=0`), `max_new_tokens=160` |
| Prompt version | `sql_v1` |
| Items | 500 held-out SQL (459 with non-empty gold) + 200 guard |
| Wall clock | 2,194 s |

### `sql_exec_holdout`

| Metric | Value |
|---|---:|
| **`exec_acc_nonempty`** (gate metric) | **0.7865** |
| `exec_acc` | 0.7860 |
| `schema_adherence` | 0.9880 |
| `valid_sql_rate` | 0.9980 |
| `exact_match_norm` | 0.5960 |
| `exec_error_rate` | 0.0120 |
| `unparseable_rate` | 0.0020 |

### `general_guard`

| Metric | Value |
|---|---:|
| `answers_in_prose` | 0.9950 |
| `non_empty_rate` | 1.0000 |
| `mean_chars` | ~225 |
| `keyword_overlap` | 0.297 |

### Reading it

**Execution accuracy exceeds exact match by 19 points** (0.786 vs 0.596). Nearly
a fifth of correct answers are written differently from the gold query. Any
evaluation of this task by string comparison under-reports by roughly that much,
which is why `exact_match_norm` is reported but never gated on.

**The base model is already competent.** 78.65% is a high bar, and it is the
right one — a gate calibrated against a weak baseline flatters everything that
follows. The `+0.05` improvement threshold means a fine-tune must reach 0.8365.

**Errors are concentrated in semantics, not syntax.** 99.8% of predictions parse
and 98.8% reference only real columns, but 21% return the wrong rows. This is not
a model that cannot write SQL; it is a model that sometimes writes the wrong
query. That shape suggests fine-tuning should help — most of the headroom is in
choosing the right aggregate, filter or column, which is exactly what in-domain
supervision teaches.

---

## 2. A measurement artefact worth 15.7 points

Recorded because it changes how the rest of this file should be read.

The first baseline run scored **0.6296**. Inspecting the failures showed a
pattern:

| Question | Gold | Prediction | Scored |
|---|---|---|---|
| Which film has a role named Taylor? | `... WHERE role = "taylor"` | `... WHERE role = 'Taylor'` | wrong |
| What was the competition in 2003? | `... WHERE year = 2003` | `... WHERE year = "2003"` | wrong |

Neither is a SQL error. The materialiser was building tables with SQLite's
defaults — case-sensitive text comparison and strict type affinity — against a
dataset written in MySQL, whose default collation is case-insensitive and whose
comparison between `'2003'` and `2003` succeeds. The harness was measuring
whether the model had guessed the dataset's capitalisation convention.

Declaring text columns `TEXT COLLATE NOCASE` and numeric columns `NUMERIC`
reproduces MySQL's semantics. The base model then scored **0.7865**.

**Why this mattered more than 15 points.** Fine-tuning on this dataset would have
taught the model to lowercase its literals, recovering most of those points
automatically. The gate would have seen a large improvement in `exec_acc` and
promoted the model, and the README would have claimed a fine-tuning gain that was
mostly a formatting convention.

Re-scoring the saved predictions with the corrected scorer took **3.5 seconds**
(`dllm eval rescore`) rather than another 36-minute inference run. Keeping
scoring separable from generation is what made fixing this cheap enough to
actually do.

---

## 3. Fine-tuned model

**NOT YET RUN.** No fine-tuning run has been executed, so no fine-tuned numbers
are claimed here.

To produce this section:

```bash
make train        # LoRA; resumable
make merge        # merge + verify against adapter-on-base
make eval         # → evals/results/finetuned.json
make compare      # regenerates the table below
make gate         # exit 1 if it must not ship
```

`make compare` writes a base-vs-fine-tuned table with per-metric deltas and
direction arrows, and appends it here.

---

## 4. Serving benchmarks

**NOT YET RUN.** The benchmark harness is complete and unit-tested, but it needs
a served model on a CUDA device, and no fine-tuned checkpoint exists yet.

```bash
docker compose up -d --build
make bench        # → artifacts/bench/latest.json
```

The sweep reports, per concurrency level: requests/s, output tokens/s, TTFT
p50/p95, end-to-end p50/p90/p95/p99, and failure count. Warm-up requests are
discarded — the first request through a cold vLLM pays CUDA graph capture and
would otherwise dominate the p99.

---

## 5. Training cost — estimated, not yet measured

From `make estimate`, using token counts measured with the real tokenizer:

| | |
|---|---:|
| Parameters | 1.54 B |
| Samples | 20,000 × 2 epochs = 40,000 |
| Optimiser steps | 2,500 (effective batch 16) |
| Tokens/example | mean 121, median 119, p95 150 |
| Total tokens | 4.82 M |
| Compute | 44.6 PFLOPs (6·N·T) |

| GPU | Est. hours | Est. $ |
|---|---:|---:|
| RTX 4060 8GB (local) | 2.58 | — |
| RTX 4090 24GB | 0.34 | 0.14 |
| L4 24GB | 0.86 | 0.64 |
| A100 40GB | 0.12 | 0.20 |
| H100 80GB | 0.07 | 0.21 |

These are **lower bounds**. At ~1,900 tokens per optimiser step the run is bound
by per-step fixed costs rather than matrix multiplies, so expect 2-4× on the
faster cards. `dllm train estimate` prints this caveat and replaces the modelled
number with a measured one after a 20-step calibration run. Detail in
[GPU_HOURS.md](GPU_HOURS.md).

---

## Reproducing everything

```bash
make setup
make audit && make data
make baseline
make estimate
```

Scorecards are committed as JSON under `evals/results/`, with per-item
predictions in the matching `.records.jsonl`, so any number above can be
recomputed or disputed without re-running inference.
