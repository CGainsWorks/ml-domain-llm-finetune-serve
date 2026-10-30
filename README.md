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

## Results

The whole pipeline has been run end to end on an RTX 4060. Every number below
was measured by a script in this repository.

### Base vs fine-tuned: 500 held-out items

| Metric | Base | Fine-tuned | Δ |
|---|---:|---:|---:|
| **`exec_acc_nonempty`** (gate metric) | 0.7865 | **0.8932** | **+0.1068** |
| `exec_acc` | 0.7860 | 0.8940 | +0.1080 |
| `exact_match_norm` | 0.5960 | 0.7960 | +0.2000 |
| `schema_adherence` | 0.9880 | 0.9960 | +0.0080 |
| `valid_sql_rate` | 0.9980 | 0.9960 | −0.0020 |
| `general_guard.answers_in_prose` | 0.9950 | 0.9950 | +0.0000 |

Correct answers: **319 → 447** of 500. **No catastrophic forgetting:** one
general question in 200 was answered with SQL, before and after.

**Gate: PROMOTE**, all six checks passed. Registered as `domain-sql` v1 with
`deployment_target` moved to it.

Fine-tuning also made inference **4× faster** on the same hardware (1,690 s →
411 s for the same 700 items): the tuned model stops at the answer instead of
continuing into explanation.

### Serving: real vLLM 0.26.0 behind the gateway, on the 4060

| Concurrency | req/s | out tok/s | TTFT p50 | e2e p50 | e2e p95 | fail |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.79 | 68.8 | 30 ms | 358 ms | 490 ms | 0 |
| 8 | 19.95 | 491.8 | 41 ms | 375 ms | 506 ms | 0 |
| 32 | 49.34 | **1,199.6** | 118 ms | 487 ms | 690 ms | 0 |

Throughput scales ~17× for 32× concurrency while p50 latency rises 36%. The
cost of concurrency lands almost entirely on time-to-first-token.

### Serving on CPU: same engine, same answers

The CPU profile (vLLM 0.26.0's CPU backend, no NVIDIA GPU) served the
fine-tuned model through the gateway and gave **identical predictions on 50 of
50** held-out items to the committed GPU scorecard. On an AVX2 desktop in
float32 it answered an 11-token query in about 4 s after a one-off 80 s warm-up,
measured with other work sharing the CPU: a fallback for development, demos and
CI, not a capacity plan.
[docs/results.md §8](docs/results.md).

### Training cost, measured

1.149 GPU-hours, 8.90 samples/s, 4.83 GB peak VRAM of 8 GB. A 20-step
calibration predicted 1.19 h, within 3.4%.

Full detail, including two measurement bugs that would have inflated these
numbers: **[docs/results.md](docs/results.md)**.

### Run once, and what has not been run

- `.github/workflows/train.yml` needs a self-hosted GPU runner, so it does not
  run on push. It was executed once, on 2026-08-10, on a self-hosted ephemeral
  runner backed by the same 4060: full 2,500-step run, merge, eval and the gate
  scoring the artefact that job produced, both jobs green. Its numbers are in
  [results.md §6](docs/results.md). The Actions log of that run did not survive
  the repository being recreated on 2026-10-15.
- The 7B profile (`configs/rented_a100.yaml`) has never been trained; it does
  not fit 8 GB.
- The trained adapter is packaged with a SHA-256 manifest but not published;
  see [below](#the-trained-adapter-packaged-not-yet-published).

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

Serving without a GPU, through the same authenticated gateway:

```bash
make demo-cpu          # vLLM's CPU backend + gateway, in containers
make smoke-serve-cpu   # the CPU serving check CI runs: auth, 429, token counters
```

## What makes the evaluation trustworthy

This is the part worth reading.

**Correctness is executed, not string-matched.** Each predicted query runs
against a SQLite database materialised from the item's own `CREATE TABLE`
schema and its result set is compared to the gold result set. `WHERE a=1 AND
b=2` and `WHERE b=2 AND a=1` are the same answer. On the base model,
`exact_match_norm` is 0.596 while `exec_acc` is 0.786: text matching
under-reports this task by 19 points.

**Rows are seeded from the gold query's literals.** Populate the table randomly
and `WHERE player = 'Jared Butler'` matches nothing; gold and prediction both
return the empty set and every wrong answer scores as correct. Seeding
guarantees the gold predicate selects a proper subset.

**The gate metric excludes empty gold results.** Same failure mode, subtler.
`exec_acc_nonempty` restricts to items where the gold query returns rows.

**A measurement artefact was found and cost 15.7 points.** The materialiser
originally used SQLite's default case-sensitive, strictly-typed comparison
against a MySQL-flavoured dataset, so `WHERE role = 'Taylor'` scored as wrong
against a gold of `WHERE role = "taylor"`. Fixing the collation and affinity to
match MySQL moved the base model from 0.6296 to 0.7865. Left in, fine-tuning
would have recovered most of those points by learning the dataset's lowercasing
convention and it would have been reported as improved SQL ability. See
[docs/MODEL_CARD.md](docs/MODEL_CARD.md#limitations).

**One definition of a prompt.** `dllm.prompts` is imported by training,
evaluation and serving. The prompt version is stamped into every scorecard and
the gate refuses to compare scorecards whose prompt versions differ, as it
refuses when item counts differ, or when there is no baseline at all.

**A catastrophic-forgetting guard.** A model that gets better at SQL and starts
answering "Who wrote Hamlet?" with `SELECT author FROM books` is blocked by
`max_regression` on `general_guard.answers_in_prose`, however good its SQL is.

**Scoring is separable from generation.** `dllm eval rescore` replays saved
per-item predictions through the current scorer: 3.5 seconds instead of 36
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

`deploy.yml` cannot reach its build job without `needs: gate` and additionally
asserts on the gate's output. A gate with no thresholds configured **fails**:
a gate that approves everything is worse than no gate, because it looks like one.

## Renting a GPU

**You probably do not need to.** The reference 1.5B config fits the 8 GB card in
roughly 2.6-6 hours. If you want it done in an hour, one A100 or L40S covers
training and both evals with room to spare.

```bash
make estimate
```

Measured inputs, not assumptions: 1.54 B parameters, 20,000 examples × 2 epochs,
**mean 121 tokens per example** (measured with the real tokenizer; assuming
`max_seq_len` would have overstated this run by 8×), 4.82 M training tokens,
44.6 PFLOPs.

Full table, the compute-bound caveat and the whole-session budget:
[docs/GPU_HOURS.md](docs/GPU_HOURS.md). Session script:
[docs/RUNBOOK_RENTED_GPU.md](docs/RUNBOOK_RENTED_GPU.md).

Training checkpoints every 50 steps and `train.resume: auto` means **re-running
the identical command is the recovery procedure**: optimiser, scheduler, RNG
and dataset position included, reattached to the same MLflow run. Spot instances
are fine.

## Serving

```bash
dllm serve keys --count 1          # plaintext key shown once; only the hash is stored
export DLLM_GATEWAY_KEYS='[{"key_id":"...","key_hash":"...","tier":"default"}]'
docker compose up --build          # GPU
make demo-cpu                      # CPU: no NVIDIA GPU needed
```

vLLM is on the internal network only: it has no meaningful authentication and
publishing its port donates the GPU. The gateway is the sole ingress and adds
bearer-token auth against hashed keys, per-key **request and token** rate limits
with `Retry-After`, a model allowlist, a prompt size cap checked before anything
reaches the GPU, Prometheus metrics with per-key token counters, and separate
`/healthz` and `/readyz`.

### Two profiles, one API

| | GPU (default) | CPU |
|---|---|---|
| Engine | `vllm/vllm-openai:v0.26.0` (CUDA) | `vllm/vllm-openai-cpu:v0.26.0` (vLLM's CPU backend) |
| Selected by | `configs/base.yaml`, `docker-compose.yml` | `configs/serve_cpu.yaml`, plus `docker-compose.cpu.yml` layered on top |
| Precision | bfloat16 | float32 (see below) |
| Needs | NVIDIA GPU and container runtime | x86-64 with AVX2 (tested; the image also ships arm64, untested). The container used about 11.4 GiB of RAM serving the 1.5B model in float32 |

Same served model name, same gateway, same OpenAI-compatible endpoints, so a
client cannot tell which profile answered. To make plain `docker compose up`
pick the CPU profile, set it in `.env`:

```bash
COMPOSE_PATH_SEPARATOR=:
COMPOSE_FILE=docker-compose.yml:docker-compose.cpu.yml
```

Bare metal: `dllm serve vllm --config configs/serve_cpu.yaml` with vLLM's CPU
build installed. The compose override and the YAML carry the same settings and
a unit test fails if they drift.

**Why vLLM's CPU backend rather than llama.cpp.** It is the same engine at the
same version, loading the same safetensors checkpoint with the same chat
template, so there is no GGUF conversion step and no second implementation of
the prompt format to drift from `dllm.prompts`. llama.cpp with a quantised GGUF
is the usual pick when CPU throughput matters most; that trade was not measured
here.

**Why float32.** On x86 without AVX-512 BF16 or AMX, vLLM's CPU backend cannot
build a oneDNN bf16 matmul, falls back to plain torch, and the first forward
pass of the 1.5B model outlasted vLLM's execute timeout and killed the engine.
float32 served it normally. On a CPU with AVX-512 BF16 or AMX,
`CPU_DTYPE=bfloat16` should work and halves the memory (not tested here).

**CI** brings the CPU profile up with the real compose files on a GitHub-hosted
runner and drives it through the gateway (`scripts/cpu_serving_smoke.sh`): 401
without a key or with a wrong one, a completion with token usage, a streamed
completion, a 429 with `Retry-After` from a low-limit key, and per-key token
counters on `/metrics`. It uses a **stand-in model**,
`HuggingFaceTB/SmolLM2-135M-Instruct` (Apache-2.0, revision pinned), because
the fine-tuned weights are not published, so CI proves the serving path and not
the SQL. Locally the same script passed against the fine-tuned model.

**On Windows this works too.** vLLM ships no Windows wheel, but it does not need
one: Docker Desktop's WSL2 backend registers an `nvidia` runtime and passes the
card through to Linux containers. Check yours before bringing the stack up:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Known gap, stated rather than hidden: rate-limit buckets are in-process, so N
replicas enforce N× the limit. Correct for one replica; horizontal scale needs
Redis.

## Metered billing via sw-rails-usage-billing

The gateway already counts what a usage bill is made of. Every successful
completion adds the prompt and completion tokens the model server reported to
`dllm_gateway_tokens_total{key_id, kind}` on `/metrics`, per API key, and every
response carries an `X-Request-Id`. Streamed responses are counted when the
client sends `stream_options: {"include_usage": true}`; without it vLLM reports
no usage, so a billing deployment should set that flag at the gateway.

[CGainsWorks/sw-rails-usage-billing](https://github.com/CGainsWorks/sw-rails-usage-billing)
(Meterly) rates usage events posted to its `POST /api/v1/events` and
`POST /api/v1/events/batch` API and deduplicates them on `transaction_id`. It is
not wired up here; this is how it would be.

| Gateway | Meterly event field |
|---|---|
| `X-Request-Id` | `transaction_id`, as `llm-<request id>` (the idempotency key) |
| `key_id` of the caller's key | `external_subscription_id`, via a lookup table (one subscription per key or per tenant) |
| `usage.prompt_tokens + usage.completion_tokens` | `properties.tokens` on a `sum` metric with code `llm_tokens` |
| served model name | `properties.model`, so one model can be priced differently from another |

The metric and plan are created once on the Meterly side, as its README shows.
The gateway then queues one event per completed request and flushes in batches:

```python
# Sketch only, not in the codebase. Called from dllm.serve.gateway.app._proxy
# after a successful upstream response, where key, request_id, body and model
# are already in scope.
import os
from datetime import datetime, timezone

import httpx

METERLY_URL = os.environ["METERLY_URL"]          # e.g. http://meterly:3000
METERLY_API_KEY = os.environ["METERLY_API_KEY"]  # an mtr_sk_ key, shown once
SUBSCRIPTIONS: dict[str, str] = {}               # key_id -> external_subscription_id

def usage_event(key_id: str, request_id: str, usage: dict, model: str) -> dict:
    return {
        "transaction_id": f"llm-{request_id}",
        "external_subscription_id": SUBSCRIPTIONS[key_id],
        "code": "llm_tokens",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "properties": {
            "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
            "model": model,
        },
    }

async def flush(client: httpx.AsyncClient, events: list[dict]) -> None:
    # Up to 500 events a request. Meterly reports on each event separately
    # (accepted, duplicate or rejected), so re-sending a whole batch after a
    # timeout cannot bill a request twice.
    for i in range(0, len(events), 500):
        r = await client.post(
            f"{METERLY_URL}/api/v1/events/batch",
            json={"events": events[i : i + 500]},
            headers={"Authorization": f"Bearer {METERLY_API_KEY}"},
            timeout=10.0,
        )
        r.raise_for_status()
```

Why per request rather than scraping the Prometheus counter: a counter resets
when the gateway restarts, is per replica, and a delta between two scrapes has
no natural idempotency key. A per-request event carries the request id, so a
replay is a `duplicate` rather than a second charge, and the counter stays
useful for dashboards and for reconciling against Meterly's own totals. Meterly
rate-limits each of its API keys to 600 requests a minute, which is why the
sketch batches.

## The trained adapter: published

The adapter behind the committed scorecard is published as the
[`adapter-v1`](https://github.com/CGainsWorks/ml-domain-llm-finetune-serve/releases/tag/adapter-v1)
release: the package archive, its manifest and `SHA256SUMS`. Check the archive
against `SHA256SUMS` before unpacking, then load it with PEFT on top of
Qwen2.5-1.5B-Instruct at revision `989aa79`. The tokenizer is not in the
package; load it from the base model at that revision, since training does not
change it.

`make package` (`dllm model package`) turns a trained adapter into a release
package under the git-ignored `artifacts/release/`: the adapter weights and
config (with the base model revision pinned), the chat template, the
Apache-2.0 licence text, a NOTICE carrying every attribution the licence audit
requires, the model card, a deterministic `.tar.gz`, `SHA256SUMS` and a JSON
manifest listing each file's name, size, SHA-256 and licence plus the training
summary and the scorecard it produced. `training_args.bin` (a pickle that also
embeds the training machine's paths) and the path-bearing run metadata are left
out deliberately.

It has been run against the adapter behind the committed scorecard (the manual
run of 2026-08-10, `evals/results/finetuned.json`):

| File | Size | SHA-256 |
|---|---:|---|
| `adapter_model.safetensors` | 73,911,112 bytes | `32a6bf05dfdbedf861faec2190bbf70ae021e774b49668752967f282434a208b` |
| `domain-sql-lora-v1.tar.gz` (whole package) | 68,471,232 bytes | `5658d0fb5c7206b8d1a8768a7f93e7a14e1ffbda97e64bc245e6c579b91c288c` |

**Licence check.** The base model, Qwen2.5-1.5B-Instruct at the pinned revision
`989aa79`, is Apache-2.0 and ungated, which permits redistributing a derivative
such as an adapter provided the licence text and a notice of changes go with
it; both are in the package. The training data is CC-BY-4.0
(`b-mc2/sql-create-context`) built from WikiSQL (BSD-3-Clause) and Spider
(CC-BY-SA-4.0), and the attributions are in the NOTICE. Whether Spider's
share-alike term reaches trained weights is not settled law; the licence audit's
position is that weights are not generally treated as an adapted work.

**Published on 2026-10-30** with that share-alike question stated in the
release notes and the NOTICE. Anyone redistributing the adapter further should
read it. The weights hash above is the one in the release; the archive hash
changes whenever the model card or NOTICE is edited and the package is rebuilt.

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

Apache-2.0, see [LICENSE](LICENSE). Dataset obligations are separate and are
recorded in [docs/DATASET_CARD.md](docs/DATASET_CARD.md).
