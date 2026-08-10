# Architecture

## The pipeline

```
                          ┌──────────────────────┐
                          │  configs/*.yaml      │
                          │  strict, _base_      │
                          │  merged, --set       │
                          └──────────┬───────────┘
                                     │
   ┌─────────────────────────────────┼─────────────────────────────────┐
   │                                 │                                 │
   ▼                                 ▼                                 ▼
┌────────────┐              ┌─────────────────┐              ┌──────────────────┐
│ LICENCE    │  blocks      │ DATA PREP       │              │ EVAL HARNESS     │
│ AUDIT      │─────────────▶│ clean, dedupe   │─────────────▶│ tasks + backends │
│            │  on failure  │ hash-split      │  train/val/  │                  │
│ downloads  │              │ verify gold     │  test/guard  │ execution-based  │
│ nothing    │              └─────────────────┘              │ scoring          │
└────────────┘                                               └────────┬─────────┘
                                                                      │
                       ┌──────────────────────────────────────────────┤
                       │                                              │
                       ▼                                              ▼
              ┌─────────────────┐                          ┌────────────────────┐
              │ BASELINE eval   │                          │ CANDIDATE eval     │
              │ (base model)    │                          │ (fine-tuned)       │
              │ runs FIRST      │                          │                    │
              └────────┬────────┘                          └─────────┬──────────┘
                       │                                             │
                       │        ┌─────────────────┐                  │
                       │        │ LoRA TRAINING   │                  │
                       │        │ resumable       │                  │
                       │        │ MLflow-tracked  │                  │
                       │        └────────┬────────┘                  │
                       │                 │                           │
                       │                 ▼                           │
                       │        ┌─────────────────┐                  │
                       │        │ MERGE + verify  │──────────────────┘
                       │        └─────────────────┘
                       │                 │
                       └────────┬────────┘
                                ▼
                       ┌─────────────────┐
                       │ PROMOTION GATE  │  min_absolute
                       │                 │  min_delta_vs_baseline
                       │ exit 0 / exit 1 │  max_regression
                       └────────┬────────┘
                          pass  │  fail
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌──────────────────────┐   ┌──────────────────────┐
        │ MLflow registry      │   │ registered + tagged  │
        │ deployment_target    │   │ "blocked"            │
        │ alias moves          │   │ alias does NOT move  │
        └──────────┬───────────┘   └──────────────────────┘
                   ▼
        ┌──────────────────────┐        ┌──────────────────────┐
        │ vLLM  (GPU)          │◀───────│ FastAPI gateway      │◀── clients
        │ OpenAI-compatible    │ internal│ auth, rate limit,   │
        │ NOT exposed          │  only  │ metrics, allowlist   │
        └──────────────────────┘        └──────────────────────┘
```

## Why it is shaped this way

### The baseline runs before the training code exists

Every claim this repository makes is a delta. A fine-tuned accuracy of 84% is
not a result; 79% → 84% on a fixed holdout with a fixed prompt is. The gate
enforces this structurally: `min_delta_vs_baseline` thresholds cause an
**error**, not a skip, when no baseline scorecard is present.

### Correctness is executed, not compared as text

The primary metric runs the predicted query against a SQLite database
materialised from the item's own schema and compares result sets. String
comparison against the gold query would mark `WHERE a=1 AND b=2` wrong for
`WHERE b=2 AND a=1`. Normalised exact match is still reported, as a lower bound.

Three things make execution comparison honest rather than convenient:

1. **Rows are seeded from the gold query's literals**, so the gold predicate
   always selects a proper subset. With randomly populated tables, gold and
   prediction both return nothing and every wrong answer scores as correct.
2. **`exec_acc_nonempty` is the gate metric.** Items whose gold returns no rows
   flatter the model for the same reason, so the gating metric excludes them.
3. **Column affinity and collation reproduce MySQL semantics**, because the
   dataset is MySQL-flavoured. Without this, `'Taylor'` vs `"taylor"` scored as
   a SQL error. Measured on the base model, that one detail was worth 15.7
   accuracy points — and fine-tuning would have "recovered" them by learning the
   dataset's capitalisation, reported as an improvement in SQL ability.

### One definition of a prompt

`dllm.prompts` is imported by training, evaluation and the merge verifier.
A prompt that differs between training and eval looks exactly like a fine-tune
that did not work, and both halves work in isolation, so nothing points at the
cause. The prompt version is written into every scorecard and the gate refuses
to compare scorecards whose prompt versions differ.

### Splits are assigned by content hash

Not by shuffling. An item's split is a pure function of its own content, so
adding a filter or changing the row cap cannot move an item from `test` into
`train` and quietly invalidate every earlier scorecard.

### Generation and scoring are separable

`dllm eval rescore` replays saved per-item records through the current scorer.
A scoring change costs seconds rather than another half hour of inference,
which is what makes it practical to fix a metric rather than live with it.

### The gate is the only path to deployment

`deploy.yml` cannot run its build job without `needs: gate`, and additionally
asserts on the gate's output. A blocked candidate is still registered in MLflow
with its scorecard and gate report attached and tagged `blocked` — so the record
of what it scored and why it was rejected survives.

### vLLM is never exposed

It has no meaningful authentication. `docker-compose.yml` puts it on the
internal network only; the gateway is the sole ingress and adds bearer-token
auth against hashed keys, per-key request *and* token rate limits, a model
allowlist and a prompt size cap checked before anything reaches the GPU.

## Module map

| Path | Responsibility |
|---|---|
| `src/dllm/config.py` | Strict typed config, `_base_` merging, `--set` overrides |
| `src/dllm/prompts.py` | The single definition of a prompt, versioned |
| `src/dllm/data/sources.py` | Dataset registry with licence and provenance metadata |
| `src/dllm/data/licence_audit.py` | Pass/fail audit; runs before any download |
| `src/dllm/data/prepare.py` | Clean, dedupe, hash-split, verify gold executability |
| `src/dllm/eval/sql/materialise.py` | `CREATE TABLE` → populated in-memory database |
| `src/dllm/eval/sql/execute.py` | Run gold and prediction, compare result sets |
| `src/dllm/eval/sql/normalise.py` | Dialect transpilation and canonicalisation |
| `src/dllm/eval/tasks/` | Task definitions and metric computation |
| `src/dllm/eval/backends/` | Local Transformers, or any OpenAI-compatible endpoint |
| `src/dllm/eval/scorecard.py` | The scorecard, and whether two are comparable |
| `src/dllm/eval/gate.py` | Thresholds → promote/block decision. Pure. |
| `src/dllm/train/lora.py` | Resumable LoRA SFT with MLflow tracking |
| `src/dllm/train/estimate.py` | GPU-hours and cost, from measured token counts |
| `src/dllm/merge/merge_adapter.py` | Merge, then verify against adapter-on-base |
| `src/dllm/registry/promote.py` | Register, then move aliases only if gated |
| `src/dllm/serve/gateway/` | Auth, rate limiting, proxy, metrics |
| `src/dllm/bench/loadgen.py` | Concurrency sweep, TTFT, latency percentiles |

## Deliberate boundaries

`requirements/base.txt` — config, data prep, the scoring maths, the gate — has
no torch and no CUDA. The eval harness, the gate and the whole unit-test suite
run on a CPU-only CI runner in under a minute. Heavy imports live inside the
factory functions that need them.
