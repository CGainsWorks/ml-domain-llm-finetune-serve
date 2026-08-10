# Model card: domain-sql

A LoRA adaptation of Qwen2.5-1.5B-Instruct for natural-language → SQL.

> **Status.** Trained, evaluated, gated, registered and served. Every number in
> this card was measured on an RTX 4060 by a script in this repository.

---

## Model details

| | |
|---|---|
| Base model | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Base revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Base licence | Apache-2.0 — commercial use permitted |
| Adaptation | LoRA, r=16, α=32, dropout 0.05 |
| Adapted modules | `q,k,v,o_proj`, `gate,up,down_proj` |
| Trainable parameters | ~0.7% of the base |
| Precision | bf16 |
| Training data | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context), CC-BY-4.0 |
| Derived from | WikiSQL (BSD-3-Clause), Spider (CC-BY-SA-4.0) |
| Output licence | Apache-2.0 for the code; see [DATASET_CARD.md](DATASET_CARD.md) for data obligations |

## Intended use

**In scope.** Generating a single SQLite/MySQL-flavoured `SELECT` query from a
natural-language question plus a `CREATE TABLE` schema supplied in the prompt.
Single-table and simple-join analytical questions over schemas that fit in the
context window.

**Out of scope.**

- **Running generated SQL against a production database without review.** The
  model has no notion of permissions, cost or data volume, and will happily
  write a query that scans everything.
- Anything other than `SELECT`. The eval harness scores a non-`SELECT`
  prediction as wrong, but nothing in the model prevents one being emitted.
- Schemas not supplied in the prompt. It has no memory of your warehouse.
- Multi-turn clarification. It answers the question it is given.
- General assistance. The guard task exists to check it has not *become* worse
  at this, not to claim it is good at it.

## Evaluation

Every number is reproducible: `make baseline` and `make eval`. The harness
executes each predicted query against a database materialised from the item's
own schema and compares result sets, so an equivalent query written differently
scores as correct. Full method in [architecture.md](architecture.md).

### Baseline — measured

Base model, no fine-tuning. 500 held-out items (459 with a non-empty gold
result), 200 guard items. Greedy decoding, prompt version `sql_v1`.

| Metric | Base model |
|---|---:|
| `exec_acc_nonempty` **(gate metric)** | **0.7865** |
| `exec_acc` | 0.7860 |
| `schema_adherence` | 0.9880 |
| `valid_sql_rate` | 0.9980 |
| `exact_match_norm` | 0.5960 |
| `exec_error_rate` | 0.0120 |
| `unparseable_rate` | 0.0020 |
| `general_guard.answers_in_prose` | 0.9950 |
| `general_guard.non_empty_rate` | 1.0000 |

Note the gap between `exec_acc` (0.786) and `exact_match_norm` (0.596). Nineteen
points of correct answers are written differently from the gold query. Any
text-match evaluation of this task under-reports by roughly that much.

### Fine-tuned — measured

LoRA r=16 on 20,000 examples for 2 epochs (1.149 GPU-hours), merged and
exported to bf16, evaluated on the identical holdout with the identical prompt.

| Metric | Base | Fine-tuned | Δ |
|---|---:|---:|---:|
| **`exec_acc_nonempty`** | 0.7865 | **0.8932** | **+0.1068** |
| `exec_acc` | 0.7860 | 0.8940 | +0.1080 |
| `exact_match_norm` | 0.5960 | 0.7960 | +0.2000 |
| `schema_adherence` | 0.9880 | 0.9960 | +0.0080 |
| `valid_sql_rate` | 0.9980 | 0.9960 | −0.0020 |
| `general_guard.answers_in_prose` | 0.9950 | 0.9950 | +0.0000 |

Correct answers 319 → 447 of 500. **Gate: PROMOTE**, all six checks passed.

Note that `exact_match_norm` rose more than `exec_acc` (+0.20 vs +0.11). Part
of what the model learned is the dataset's *formatting* conventions, not just
its semantics. Execution accuracy is the honest number because it does not care
how a query is written, and it is the one the gate uses.

### Serving

Real vLLM 0.26.0 on the same 8 GB card, behind the gateway: peak **1,199.6
output tokens/s** at concurrency 32, TTFT p50 118 ms, e2e p50 487 ms, zero
failures. Full sweep in [results.md](results.md#4-serving-benchmarks).

### Promotion thresholds

Configured in `configs/base.yaml`. A model that fails any of these is not
deployable, and `deploy.yml` cannot reach its build job without a passing gate.

| Kind | Metric | Threshold |
|---|---|---:|
| absolute floor | `sql_exec_holdout.valid_sql_rate` | ≥ 0.98 |
| absolute floor | `sql_exec_holdout.schema_adherence` | ≥ 0.95 |
| absolute floor | `general_guard.non_empty_rate` | ≥ 0.98 |
| improvement | `sql_exec_holdout.exec_acc_nonempty` | ≥ +0.05 vs baseline |
| regression cap | `general_guard.answers_in_prose` | ≤ 0.10 drop |
| regression cap | `sql_exec_holdout.valid_sql_rate` | ≤ 0.01 drop |

With the baseline at 0.7865, a candidate must reach **0.8365** to ship.

---

## Limitations

Stated plainly, because a model card that lists only strengths is marketing.

### The evaluation

- **The holdout is in-distribution.** It is a genuine holdout of
  `sql-create-context` — assigned by content hash, deduplicated before
  splitting, never trained on. It is *not* an out-of-distribution test. The
  dataset partly derives from Spider, so scores here will overstate performance
  on an unfamiliar schema style.
- **Schemas are shallow.** Mostly single tables of `VARCHAR` columns with no
  foreign keys, indexes or realistic types. Real warehouse schemas are harder
  in ways this benchmark cannot see.
- **Synthetic rows.** Tables are populated with 12 deterministic rows seeded
  from the gold query's literals. This makes the metric well-posed but means
  it never tests behaviour on large tables, NULL handling, duplicate rows or
  realistic value distributions.
- **Execution accuracy still accepts coincidences.** Two different queries can
  return the same result set over 12 rows. `exec_acc_nonempty` reduces this by
  excluding empty gold results, but does not eliminate it.
- **One reading of an ambiguous question.** A query that is correct under a
  different reasonable interpretation is scored wrong.
- **Measurement artefacts are a real risk, and one was found.** The materialiser
  originally used SQLite's default (case-sensitive, strict-typed) comparison
  against a MySQL-flavoured dataset. That scored `WHERE role = 'Taylor'` as
  wrong against a gold of `WHERE role = "taylor"`. Correcting the collation and
  affinity to match MySQL moved the base model from 0.6296 to **0.7865** — 15.7
  points. Had it not been caught, fine-tuning would have recovered most of those
  points by learning the dataset's lowercasing convention, and it would have
  been reported as improved SQL ability. Other artefacts of this kind may
  remain.

### The model

- 1.5B parameters. Complex joins, window functions, correlated subqueries and
  CTEs are where it will fail first.
- Trained on ≤512-token sequences. Long schemas degrade it.
- Greedy decoding is assumed throughout. Sampled output is not evaluated and
  will be worse.
- No SQL injection or safety reasoning of any kind. Treat generated SQL as
  untrusted input to your database.

### The serving stack

- **Rate limiting is in-process.** Two gateway replicas each enforce the limit
  independently, so the effective limit is N× the configured one. Correct for a
  single replica; horizontal scale needs the buckets moved to Redis. This is a
  known gap, not an oversight.
- **The API-key store is loaded at startup.** Revoking a key requires a
  restart.
- The token-based rate limit charges an estimate before generation and refunds
  the difference afterwards. A caller can briefly exceed the token budget within
  one request.
- No per-tenant isolation, quotas or billing beyond the rate limiter.

### Provenance and licensing

- Training data derives partly from **Spider (CC-BY-SA-4.0)**. This repository
  ships the preparation script rather than the prepared splits, so it does not
  redistribute a derivative dataset. If you redistribute the splits, the
  share-alike term applies to them.
- Attribution is owed to b-mc2, Salesforce (WikiSQL) and Yale LILY (Spider).
- **No PII review was performed** beyond noting that the source consists of
  synthetic and public sports/geography style tables.
- Base model weights are Apache-2.0; a LoRA adapter over them inherits no
  additional restriction.

## Environmental and compute cost

| | |
|---|---|
| Baseline eval | 1,690 s on one RTX 4060 (8 GB) |
| Fine-tuned eval | 411 s on the same card |
| Training | **1.149 GPU-hours measured**, 4.83 GB peak VRAM |
| Total compute | ~45 PFLOPs for the reference training run |

Small by current standards, which is the point: the interesting engineering here
is the evaluation gate, not the scale. See [GPU_HOURS.md](GPU_HOURS.md).

## Reproducing

```bash
make setup
make audit      # licence audit; downloads nothing; exits non-zero on a violation
make data
make baseline   # the numbers in the baseline table above
make train
make merge
make eval
make compare    # writes docs/results.md
make gate       # exit 1 if this model must not ship
```

## Citation

```bibtex
@software{ml_domain_llm_finetune_serve,
  title  = {ml-domain-llm-finetune-serve: eval-gated LoRA fine-tuning and serving},
  author = {CGainsWorks},
  year   = {2026},
  url    = {https://github.com/CGainsWorks/ml-domain-llm-finetune-serve}
}
```

Base model: Qwen2.5 (Qwen Team, 2024). Data: b-mc2 `sql-create-context`, from
WikiSQL (Zhong et al., 2017) and Spider (Yu et al., 2018).
