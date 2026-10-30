# Results

Every number here was measured, on the hardware named, by a script in this
repository. Nothing is estimated unless it says so.

**Headline: execution accuracy went from 0.7865 to 0.8932 (+0.1068) on a
500-item holdout, with no regression on the catastrophic-forgetting guard. The
model passed the promotion gate and was registered.**

---

## 1. Base vs fine-tuned

```bash
make baseline    # → evals/results/baseline.json
make train && make merge
make eval        # → evals/results/finetuned.json
make compare
```

| | Base | Fine-tuned |
|---|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798` | + LoRA r=16, merged |
| Hardware | 1 × RTX 4060 (8 GB), bf16 | same |
| Decoding | greedy, `max_new_tokens=160` | same |
| Prompt version | `sql_v1` | `sql_v1` |
| Eval wall clock | 1,690 s | 411 s |

### `sql_exec_holdout` — 500 held-out items, 459 with non-empty gold

| Metric | Base | Fine-tuned | Δ | |
|---|---:|---:|---:|:--:|
| **`exec_acc_nonempty`** (gate metric) | 0.7865 | **0.8932** | **+0.1068** | ▲ |
| `exec_acc` | 0.7860 | 0.8940 | +0.1080 | ▲ |
| `exact_match_norm` | 0.5960 | 0.7960 | +0.2000 | ▲ |
| `schema_adherence` | 0.9880 | 0.9960 | +0.0080 | ▲ |
| `valid_sql_rate` | 0.9980 | 0.9960 | −0.0020 | ▼ |
| `exec_error_rate` | 0.0120 | 0.0060 | −0.0060 | ▲ |
| `unparseable_rate` | 0.0020 | 0.0040 | +0.0020 | ▼ |

Raw counts: correct 319 → **447**, wrong 174 → **48**, exec errors 6 → 3.
Zero items had an unexecutable gold query in either run, so the denominators are
identical and the comparison is like-for-like.

### `general_guard` — 200 general questions, never trained on

| Metric | Base | Fine-tuned | Δ | |
|---|---:|---:|---:|:--:|
| **`answers_in_prose`** (guard metric) | 0.9950 | 0.9950 | +0.0000 | = |
| `non_empty_rate` | 1.0000 | 1.0000 | +0.0000 | = |
| `mean_chars` | 232.94 | 214.00 | −18.94 | ▼ |
| `keyword_overlap` | 0.2997 | 0.2946 | −0.0051 | ▼ |

**No catastrophic forgetting.** Exactly one of 200 general questions was
answered with SQL, before and after. Answers got about 8% shorter, which is the
expected pull of a terse-output fine-tune and nowhere near the collapse the
guard exists to catch.

### Reading it

**The gain is real, not a formatting artefact.** `exact_match_norm` rose more
than `exec_acc` (+0.20 vs +0.11), which is what learning the dataset's
conventions looks like — but execution accuracy rose independently, and that
metric does not care how a query is written. See §3 for why this distinction
was nearly lost.

**Errors were already syntactic-clean before tuning and stayed that way.**
99.8% → 99.6% valid SQL. The headroom was never in producing parseable SQL; it
was in choosing the right filter, aggregate or column, and that is what moved.

**Fine-tuning made inference 4× faster on the same hardware** (1,690 s → 411 s
for the same 700 items). The tuned model stops at the answer instead of
continuing into explanation, so it hits the stop strings far sooner. That is a
serving-cost result as much as an accuracy one.

---

## 2. Promotion gate

```bash
make gate     # exit 0
```

**Decision: PROMOTE.** All six checks passed.

| | Check | Metric | Threshold | Baseline | Observed |
|:--:|---|---|---:|---:|---:|
| ✅ | min_absolute | `general_guard.non_empty_rate` | ≥ 0.9800 | — | 1.0000 |
| ✅ | min_absolute | `sql_exec_holdout.schema_adherence` | ≥ 0.9500 | — | 0.9960 |
| ✅ | min_absolute | `sql_exec_holdout.valid_sql_rate` | ≥ 0.9800 | — | 0.9960 |
| ✅ | min_delta | `sql_exec_holdout.exec_acc_nonempty` | ≥ +0.0500 | 0.7865 | 0.8932 |
| ✅ | max_regression | `general_guard.answers_in_prose` | ≤ 0.1000 drop | 0.9950 | 0.9950 |
| ✅ | max_regression | `sql_exec_holdout.valid_sql_rate` | ≤ 0.0100 drop | 0.9980 | 0.9960 |

Registered as `domain-sql` version 1; `Staging` and `deployment_target` both
moved to it. The same command run against a non-improving candidate exits 1,
tags the version `blocked`, and leaves `deployment_target` where it was.

---

## 3. A measurement artefact worth 15.7 points

Recorded because it changes how the rest of this file should be read.

The first baseline scored **0.6296**. Inspecting the failures showed a pattern:

| Question | Gold | Prediction | Scored |
|---|---|---|---|
| Which film has a role named Taylor? | `... WHERE role = "taylor"` | `... WHERE role = 'Taylor'` | wrong |
| What was the competition in 2003? | `... WHERE year = 2003` | `... WHERE year = "2003"` | wrong |

Neither is a SQL error. The materialiser was building tables with SQLite's
defaults — case-sensitive text comparison, strict type affinity — against a
dataset written in MySQL, whose default collation is case-insensitive and whose
`'2003' = 2003` comparison succeeds. The harness was measuring whether the model
had guessed the dataset's capitalisation convention.

Declaring text columns `TEXT COLLATE NOCASE` and numeric columns `NUMERIC`
reproduces MySQL semantics. The base model then scored **0.7865**.

**Why this mattered more than the points.** Fine-tuning teaches exactly that
capitalisation convention. Those 15.7 points would have come back automatically,
the gate would have seen a large `exec_acc` gain, and this file would have
claimed a fine-tuning improvement that was mostly a formatting habit. The real
gain, measured against a correct baseline, is +0.1068 — large, but not the
+0.26 the broken harness would have reported.

Re-scoring the saved predictions with the corrected scorer took **3.5 seconds**
(`dllm eval rescore`) rather than another 28-minute inference run. Keeping
scoring separable from generation is what made fixing this cheap enough to do.

---

## 4. Serving benchmarks

```bash
docker compose up -d --build
make bench    # → artifacts/bench/latest.json
```

Real vLLM 0.26.0 on the RTX 4060, behind the authenticated FastAPI gateway.
Streaming, `max_tokens=128`, 48 requests per level, warm-up discarded.

| Concurrency | req/s | out tok/s | TTFT p50 | TTFT p95 | e2e p50 | e2e p95 | e2e p99 | fail | 429 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.79 | 68.8 | 30 ms | 34 ms | 358 ms | 490 ms | 669 ms | 0 | 0 |
| 2 | 5.41 | 133.4 | 39 ms | 44 ms | 357 ms | 505 ms | 715 ms | 0 | 0 |
| 4 | 10.23 | 252.1 | 41 ms | 56 ms | 383 ms | 520 ms | 690 ms | 0 | 0 |
| 8 | 19.95 | 491.8 | 41 ms | 67 ms | 375 ms | 506 ms | 692 ms | 0 | 0 |
| 16 | 33.55 | 826.9 | 54 ms | 90 ms | 411 ms | 558 ms | 706 ms | 0 | 0 |
| 32 | 49.34 | 1,199.6 | 118 ms | 214 ms | 487 ms | 690 ms | 889 ms | 0 | 0 |

**Peak: 1,199.6 output tokens/s at concurrency 32**, p50 487 ms, p95 690 ms,
TTFT p50 118 ms.

### Reading it

**Throughput scales ~17× for 32× concurrency** while p50 latency rises only 36%
(358 → 487 ms). That is continuous batching working: the card is latency-bound
at low concurrency and only starts trading latency for throughput past 16.

**TTFT is the metric that degrades**, 30 ms → 118 ms, because requests begin
queueing for a scheduler slot. End-to-end p50 barely moves, so the cost of
concurrency lands almost entirely on time-to-first-token — which is what a user
perceives as responsiveness.

**These are 8 GB numbers.** `gpu_memory_utilization=0.80` on a card holding a
2.9 GB checkpoint leaves a small KV cache, capping useful concurrency. A 24 GB
card would push the knee well past 32. Do not read these as representative of
datacentre hardware.

### The first benchmark run was invalid, and said so

The initial sweep reported 273 of 288 requests failed. They were **429s from our
own gateway**: the load generator was firing 48 requests per level against a
60 rpm default limit, so the benchmark was measuring the rate limiter.

Two fixes, both kept: `dllm serve keys` gained `--rpm` and `--tokens-per-min`
so a benchmarking key can be minted with raised limits, and the harness now
counts 429s separately, refuses to call the run `valid`, and prints the exact
command to fix it. A benchmark that silently reports the limiter's throughput as
the model's is worse than one that fails.

---

## 5. Training cost — measured

| | |
|---|---:|
| Parameters | 1.54 B (18.5 M trainable, 1.18%) |
| Samples | 20,000 × 2 epochs = 40,000 |
| Optimiser steps | 2,500 (effective batch 16) |
| Tokens/example | mean 121, median 119, p95 150 |
| Total tokens | 4.82 M |
| **Wall clock** | **4,135 s (1.149 GPU-hours)** |
| **Throughput** | **8.90 samples/s** |
| **Peak VRAM** | **4.83 GB of 8 GB** |
| Final train loss | 0.0294 |

The 20-step calibration predicted **1.19 h**; the run took **1.149 h** — within
3.4%. The pre-calibration analytic estimate said 2.58 h, because it assumed an
MFU of 0.16 for this card. Measured MFU is 0.35, and `configs/gpus.yaml` now
carries the measurement. See [GPU_HOURS.md](GPU_HOURS.md).

Full session on the 4060, end to end: ~2.5 hours including both evals.

---

## 6. Run-to-run variance, and what the gate threshold is worth

The pipeline has been trained twice from scratch on the same card with the same
seed: once by hand, once through `train.yml` on a self-hosted GitHub Actions
runner. Independent checkouts, independent virtualenvs.

| | Manual run | CI run (`train.yml`) | Δ |
|---|---:|---:|---:|
| `exec_acc_nonempty` | 0.8932 | 0.8954 | +0.0022 |
| `schema_adherence` | 0.9960 | 0.9980 | +0.0020 |
| `valid_sql_rate` | 0.9960 | 0.9980 | +0.0020 |
| Final train loss | 0.029406 | 0.029349 | −0.000057 |
| Wall clock | 4,135 s | 4,103 s | −32 s |
| Peak VRAM | 4.826 GB | 4.826 GB | 0 |

**Seeding does not buy bitwise determinism on a GPU.** Everything seedable is
seeded — data order, shuffling, dropout, init — but cuDNN kernel selection and
floating-point reduction order are not deterministic by default, so two runs of
identical code diverge slightly and the divergence compounds over 2,500 steps.

This is the number that makes the gate meaningful: **the noise floor on the gate
metric is about ±0.002, and the promotion threshold is +0.05 — roughly 23× it.**
A gate set at +0.002 would fire on coin flips. It is also why the 15.7-point
measurement artefact in §3 mattered so much: it was 70× the noise, sitting
inside a metric nobody was checking.

If you need bitwise reproducibility rather than statistical, set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and call
`torch.use_deterministic_algorithms(True)`. It costs throughput and is not on by
default here, because knowing the variance is more useful than pretending it is
zero.

---

## 7. Merge fidelity

`make merge` refuses to export a checkpoint that does not reproduce
adapter-on-base behaviour. Measured on this run:

| Check | Argmax agreement | max \|Δlogit\| |
|---|---:|---:|
| Merge fidelity (fp32 vs fp32) | **1.000000** | 0.000253 |
| Export cast (bf16 vs fp32) | 0.962025 | 1.871325 |

The merge is numerically exact. Casting the result to bf16 for export moves
3.8% of next-token choices — which is what reduced precision does, and is why
only the first figure is gated. The eval above was run on the exported bf16
checkpoint, so that cost is already inside the reported accuracy.

---

## 8. CPU serving: same engine, same answers

`configs/serve_cpu.yaml` and `docker-compose.cpu.yml` serve the merged model on
vLLM 0.26.0's CPU backend (`vllm/vllm-openai-cpu:v0.26.0`) behind the same
gateway. Measured on 2026-10-30 on the development desktop: x86-64 with AVX2
and no AVX-512, Docker Desktop on WSL2, vLLM binding 7 OpenMP threads.

**Parity.** The first 50 holdout items, sent through the gateway to the CPU
profile in float32, against the committed GPU scorecard
(`evals/results/finetuned.json`, bf16 on the 4060, Hugging Face backend):

| | CPU profile via gateway | Committed GPU run, same 50 items |
|---|---:|---:|
| Identical predictions | **50 of 50** | |
| Correct | 45 | 45 |
| Wrong result | 4 | 4 |
| Execution error | 1 | 1 |
| `exec_acc_nonempty` (46 items) | 0.9130 | 0.9130 |

The exported checkpoint is bf16, so float32 serving computes on exactly the
same weights at higher precision. Scorecard and per-item records:
`evals/results/cpu_served_sample.json` and `.records.jsonl`. Fifty items is a
parity check, not a new accuracy figure; the 500-item numbers in §1 remain the
reference.

**Latency, float32, weights on a Docker named volume.** Other containers were
using the same CPU at the time, so these are indicative, not a benchmark:

| | |
|---|---:|
| Container start to `/health` OK | 82 s |
| First request (one-off warm-up) | 80 s |
| Next requests, 11-token SQL answer | about 4 s each |

The same model on a bind mount from the Windows filesystem took more than 5
minutes for the first request and 13 to 19 s for the next ones, so on Docker
Desktop for Windows keep the weights in a named volume or inside WSL.

**bfloat16 does not work on this CPU.** Without AVX-512 BF16 or AMX, vLLM's CPU
backend logs `Failed to create oneDNN linear, fallback to torch linear` and the
first forward pass of the 1.5B model ran past vLLM's 300 s execute timeout and
killed the engine. That happened twice, on a bind mount and on a named volume.
Hence float32 as the profile default, a 900 s execute timeout for headroom and
a container health check that only reports healthy after one real completion
has been served.

**In CI** the same compose files run on a GitHub-hosted runner with a stand-in
model (`HuggingFaceTB/SmolLM2-135M-Instruct`, Apache-2.0, revision pinned),
because the fine-tuned weights are not published. That job checks the serving
path (engine, OpenAI API, auth, rate limit, token accounting), not SQL quality.

---

## Reproducing

```bash
make setup
make audit && make data
make estimate
make baseline
make train && make merge && make eval && make compare && make gate
docker compose up -d --build && make bench
```

Scorecards are committed as JSON under `evals/results/`, with per-item
predictions in the matching `.records.jsonl`, so any number above can be
recomputed — or disputed — without re-running inference.
