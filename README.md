# ml-domain-llm-finetune-serve

**A domain LLM pipeline where the evaluation gate is the point.** Licence audit →
dataset preparation → *baseline first* → LoRA fine-tuning → merge → execution-based
evaluation → a promotion gate that CI cannot route around → vLLM behind an
authenticated FastAPI gateway.

```
 licence audit ──▶ data prep ──▶ BASELINE eval ──▶ LoRA train ──▶ merge
       │                              │                             │
   blocks the                    the before-number             ┌────┴────┐
   build on a                    everything else is            │ EVAL    │
   bad licence                   measured against              └────┬────┘
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │ PROMOTION GATE   │
                                                          │ exit 0 / exit 1  │
                                                          └────┬────────┬────┘
                                                         pass  │        │ fail
                                                               ▼        ▼
                                                   registry +      registered,
                                                   deploy          tagged "blocked",
                                                                   NOT deployed
```

Full diagram and rationale: [docs/architecture.md](docs/architecture.md).

---

## Status

| Stage | State |
|---|---|
| Licence audit + dataset card | **Done**, runs in CI, blocks on violation |
| Dataset preparation (78,577 → 20,000/500/500/200) | **Done** |
| Eval harness (execution-based) | **Done**, 204 unit tests |
| **Baseline scorecard** | **Done and committed** — see below |
| LoRA training, resumable, MLflow | **Done**, verified end to end on CPU |
| Adapter merge + verification | **Done**, verified |
| Promotion gate | **Done**, 20 unit tests |
| MLflow registry promotion | **Done** (SQLite backend) |
| vLLM + FastAPI gateway | **Done**; not yet run against a real GPU |
| Benchmarks | **Harness done**; no numbers yet — needs a served model |
| CI/CD with eval-gated deploy | **Done** |
| **Fine-tuning run** | **Not yet run.** No fine-tuned numbers are claimed. |

Nothing in this repository quotes a number that was not measured. Where a number
does not exist yet, it says so.

## The baseline

Qwen2.5-1.5B-Instruct, no fine-tuning, 500 held-out items + 200 guard items,
greedy decoding, 2,194 s on one RTX 4060.

| Metric | Value |
|---|---:|
| **`exec_acc_nonempty`** (the gate metric) | **0.7865** |
| `exec_acc` | 0.7860 |
| `schema_adherence` | 0.9880 |
| `valid_sql_rate` | 0.9980 |
| `exact_match_norm` | 0.5960 |
| `general_guard.answers_in_prose` | 0.9950 |

Reproduce: `make baseline`. Raw scorecard: [`evals/results/baseline.json`](evals/results/baseline.json).

A fine-tune must reach **0.8365** to clear the gate.

## Quickstart

```bash
make setup        # venv + CUDA torch + everything
make audit        # licence audit. Downloads nothing. Exits 1 on a violation.
make data         # 78,577 rows → clean, dedupe, hash-split, verify gold
make estimate     # GPU hours and cost, from measured token counts
make baseline     # evaluate the BASE model. Before any training.
make train        # LoRA. Re-run the same command to resume after a crash.
make merge && make eval && make compare
make gate         # exit 0 = deployable, exit 1 = do not ship
```

No GPU? The eval harness, the gate and the whole test suite are pure Python:

```bash
make setup-cpu && make test
```

Whole pipeline end to end in two minutes on CPU (what CI runs):

```bash
make smoke
```

## What makes the evaluation trustworthy

This is the part worth reading.

**Correctness is executed, not string-matched.** Each predicted query runs
against a SQLite database materialised from the item's own `CREATE TABLE`
schema, and its result set is compared to the gold result set. `WHERE a=1 AND
b=2` and `WHERE b=2 AND a=1` are the same answer. On the base model,
`exact_match_norm` is 0.596 while `exec_acc` is 0.786 — text matching
under-reports this task by 19 points.

**Rows are seeded from the gold query's literals.** Populate the table randomly
and `WHERE player = 'Jared Butler'` matches nothing; gold and prediction both
return the empty set, and every wrong answer scores as correct. Seeding
guarantees the gold predicate selects a proper subset.

**The gate metric excludes empty gold results.** Same failure mode, subtler.
`exec_acc_nonempty` restricts to items where the gold query returns rows.

**A measurement artefact was found and cost 15.7 points.** The materialiser
originally used SQLite's default case-sensitive, strictly-typed comparison
against a MySQL-flavoured dataset, so `WHERE role = 'Taylor'` scored as wrong
against a gold of `WHERE role = "taylor"`. Fixing the collation and affinity to
match MySQL moved the base model from 0.6296 to 0.7865. Left in, fine-tuning
would have recovered most of those points by learning the dataset's lowercasing
convention — and it would have been reported as improved SQL ability. See
[docs/MODEL_CARD.md](docs/MODEL_CARD.md#limitations).

**One definition of a prompt.** `dllm.prompts` is imported by training,
evaluation and serving. The prompt version is stamped into every scorecard, and
the gate refuses to compare scorecards whose prompt versions differ — as it
refuses when item counts differ, or when there is no baseline at all.

**A catastrophic-forgetting guard.** A model that gets better at SQL and starts
answering "Who wrote Hamlet?" with `SELECT author FROM books` is blocked by
`max_regression` on `general_guard.answers_in_prose`, however good its SQL is.

**Scoring is separable from generation.** `dllm eval rescore` replays saved
per-item predictions through the current scorer — 3.5 seconds instead of 36
minutes. That is what makes it practical to fix a metric rather than live with
one you know is wrong.

## The gate

`configs/base.yaml`:

| Kind | Metric | Threshold |
|---|---|---:|
| floor | `sql_exec_holdout.valid_sql_rate` | ≥ 0.98 |
| floor | `sql_exec_holdout.schema_adherence` | ≥ 0.95 |
| floor | `general_guard.non_empty_rate` | ≥ 0.98 |
| improvement | `sql_exec_holdout.exec_acc_nonempty` | ≥ +0.05 vs baseline |
| regression cap | `general_guard.answers_in_prose` | ≤ 0.10 drop |
| regression cap | `sql_exec_holdout.valid_sql_rate` | ≤ 0.01 drop |

`deploy.yml` cannot reach its build job without `needs: gate`, and additionally
asserts on the gate's output. A gate with no thresholds configured **fails** —
a gate that approves everything is worse than no gate, because it looks like one.

## Renting a GPU

**You probably do not need to.** The reference 1.5B config fits the 8 GB card in
roughly 2.6-6 hours. If you want it done in an hour, one A100 or L40S covers
training and both evals with room to spare.

```bash
make estimate
```

Measured inputs, not assumptions: 1.54 B parameters, 20,000 examples × 2 epochs,
**mean 121 tokens per example** (measured with the real tokenizer — assuming
`max_seq_len` would have overstated this run by 8×), 4.82 M training tokens,
44.6 PFLOPs.

Full table, the compute-bound caveat, and the whole-session budget:
[docs/GPU_HOURS.md](docs/GPU_HOURS.md). Session script:
[docs/RUNBOOK_RENTED_GPU.md](docs/RUNBOOK_RENTED_GPU.md).

Training checkpoints every 50 steps and `train.resume: auto` means **re-running
the identical command is the recovery procedure** — optimiser, scheduler, RNG
and dataset position included, reattached to the same MLflow run. Spot instances
are fine.

## Serving

```bash
dllm serve keys --count 1          # plaintext key shown once; only the hash is stored
export DLLM_GATEWAY_KEYS='[{"key_id":"...","key_hash":"...","tier":"default"}]'
docker compose up --build
```

vLLM is on the internal network only — it has no meaningful authentication, and
publishing its port donates the GPU. The gateway is the sole ingress and adds
bearer-token auth against hashed keys, per-key **request and token** rate limits
with `Retry-After`, a model allowlist, a prompt size cap checked before anything
reaches the GPU, Prometheus metrics, and separate `/healthz` and `/readyz`.

Known gap, stated rather than hidden: rate-limit buckets are in-process, so N
replicas enforce N× the limit. Correct for one replica; horizontal scale needs
Redis.

## Data and licensing

- Code: Apache-2.0.
- Training data: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)
  (CC-BY-4.0), from WikiSQL (BSD-3-Clause) and **Spider (CC-BY-SA-4.0)**.
- Base model: Qwen2.5-1.5B-Instruct, Apache-2.0, revision-pinned.
- This repository ships the **preparation script, not the prepared splits**, so
  it does not redistribute a share-alike derivative dataset.
- `dllm data audit` runs before any download and fails the build on a
  non-commercial or undeclared licence. The registry deliberately keeps a
  `REJECTED` entry (Alpaca, CC-BY-NC-4.0) so the audit's teeth are testable.

[docs/DATASET_CARD.md](docs/DATASET_CARD.md) ·
[docs/LICENCE_AUDIT.md](docs/LICENCE_AUDIT.md)

## Documentation

| | |
|---|---|
| [architecture.md](docs/architecture.md) | Diagram, design decisions, module map |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use and an honest limitations section |
| [DATASET_CARD.md](docs/DATASET_CARD.md) | Generated from the registry and prep stats |
| [LICENCE_AUDIT.md](docs/LICENCE_AUDIT.md) | Generated; provenance and obligations |
| [GPU_HOURS.md](docs/GPU_HOURS.md) | What to rent and for how long |
| [RUNBOOK_RENTED_GPU.md](docs/RUNBOOK_RENTED_GPU.md) | Session script, failure modes |
| [results.md](docs/results.md) | Base vs fine-tuned scorecard and benchmarks |

## Limitations

The holdout is in-distribution; schemas are shallow single tables; rows are
synthetic; the model is 1.5B and will fail first on window functions, CTEs and
complex joins; rate limiting does not survive horizontal scaling. Written out
properly in [docs/MODEL_CARD.md](docs/MODEL_CARD.md#limitations).

## Licence

Apache-2.0 — see [LICENSE](LICENSE). Dataset obligations are separate and are
recorded in [docs/DATASET_CARD.md](docs/DATASET_CARD.md).
