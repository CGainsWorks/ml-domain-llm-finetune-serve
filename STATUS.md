# Status

Last updated: 2026-08-10

## Verified by running it

| Component | Evidence |
|---|---|
| Licence audit | Passes on the default mix; **blocks** the deliberately-registered non-commercial source with exit 1. |
| Dataset preparation | 78,577 raw rows → 20,000 train / 500 validation / 500 test / 200 guard. 193 unparseable and 86 non-executable gold queries dropped. |
| Eval harness | 275 unit tests. Run against the real base model three times. |
| **Baseline scorecard** | `evals/results/baseline.json` — 500 + 200 items, 1,690 s on an RTX 4060. |
| LoRA training | Full 2,500-step run on the 4060. Calibrated at 9.35 samples/s, 4.6 GB peak VRAM. |
| Resume from checkpoint | Killed and re-run; resumed from `checkpoint-8` and reattached to the **same** MLflow run id. |
| Adapter merge | Merged and verified against adapter-on-base outputs before export. |
| Promotion gate | Blocks a candidate equal to the baseline (delta 0 < required 0.05), exit 1, report naming the failing check. |
| MLflow registry | `registry promote` registered version 1, tagged it `blocked`, and left `deployment_target` **unset**. Verified via the client. |
| FastAPI gateway | 17 HTTP tests plus the real container: 401 no token, 401 wrong token, 502 valid token (auth passed, upstream absent), 400 disallowed model, 503 with no keys configured. Runs as uid 10001, not root. |
| Gateway Docker image | Built and run locally; `/healthz`, `/readyz`, `/metrics` all correct. |
| Benchmark harness | 15 tests against a stub OpenAI endpoint: SSE parsing, TTFT, percentiles, failure handling. |
| **End-to-end smoke** | `scripts/run_smoke.sh` runs all 7 stages and asserts the gate blocks the resulting model. Passes locally **and on a clean GitHub Actions runner**. |
| **CI** | All 6 jobs green on Ubuntu: lint + tests on 3.10 and 3.12, licence audit, config validation, image build, end-to-end smoke. |

## Complete

The pipeline has been run end to end on an RTX 4060: trained, merged,
evaluated, gated, registered, served and benchmarked.

- **exec_acc_nonempty 0.7865 → 0.8932** (+0.1068) on 500 held-out items.
- **Gate: PROMOTE.** Registered as `domain-sql` v1, `deployment_target` moved.
- **Training: 1.149 GPU-hours measured**, against a 1.19 h calibration.
- **Serving: 1,199.6 output tokens/s peak** at concurrency 32, zero failures,
  real vLLM 0.26.0 behind the gateway.

Numbers and their caveats: [docs/results.md](docs/results.md).

## Not done

- `.github/workflows/train.yml` needs a self-hosted GPU runner and has not been
  executed on one.
- The 7B profile (`configs/rented_a100.yaml`) has never been trained. It does
  not fit 8 GB and needs the rented card the brief anticipated.

### vLLM on Windows: it does run

An earlier version of this file claimed the serving stack could not be verified
here because vLLM publishes no Windows wheel. That is true of `pip install` and
irrelevant to running it. Docker Desktop's WSL2 backend registers an `nvidia`
container runtime and passes the card straight through, so the real vLLM image
runs against the local RTX 4060. Confirmed on driver 591.86:

```
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
-> NVIDIA GeForce RTX 4060, 8188 MiB
```

No substitute inference server is needed, and none is used: the benchmarks will
measure the same vLLM that `docker compose up` runs in production.

## What running it actually caught

Every one of these was found by executing the pipeline, not by writing tests
against it. They are listed because the list is the argument for doing so.

1. **A measurement artefact worth 15.7 accuracy points.** The harness built its
   verification databases with SQLite's case-sensitive, strictly-typed defaults
   against a MySQL-flavoured dataset, so `WHERE role = 'Taylor'` scored wrong
   against a gold of `WHERE role = "taylor"`. Fixing the collation and affinity
   moved the base model from 0.6296 to 0.7865. Left in, fine-tuning would have
   recovered most of those points by learning the dataset's lowercasing
   convention and it would have been reported as improved SQL ability.
2. **An unterminated string literal killed a 35-minute run.** `sqlglot` raises
   `TokenError` from the lexer, not `ParseError`, and it propagated out of
   scoring *after* all 500 generations had completed. Scoring can no longer
   raise.
3. **CI could never pass its own staleness check.** The generated licence audit
   embedded a timestamp, so `git diff` always reported a change.
4. **`.gitattributes` marked scorecards binary.** `-diff` meant `git diff`
   showed `Bin 2134 -> 2163 bytes`, destroying the reason for committing them.
5. **The smoke run overwrote the committed dataset card** with its 64-example
   numbers. Isolating the data directory was not enough; the docs output needed
   redirecting too.
6. **The gateway crashed on a second app instance.** Prometheus metrics were
   registered on the global default registry.
7. **The CLI crashed printing its own scorecard** on a Windows console, because
   the comparison table contains `Δ` and cp1252 cannot encode it. This killed the
   first full smoke run at the last step.
8. **The 4060 estimate was 2.2x pessimistic.** Measured MFU is 0.35, not the
   0.16 assumed, so the projection went from 2.58 h to 1.18 h.
9. **The heartbeat ETA was 2.4x wrong early in a run**, because the rate was
   cumulative and included model loading.
10. **`make merge` would have merged the wrong adapter.** The Makefile rebuilt
    the checkpoint path with a shell `yaml.safe_load` that skips `_base_`
    resolution.
11. **The merge check rejected a numerically exact fine-tune.** It compared
    decoded greedy text, which is chaotic in the logits. Its replacement then
    compared fp32 against bf16 and blamed the merge for the cast. Merge fidelity
    and export precision are now measured separately: 1.000000 and 0.962025.
12. **vLLM would not start under WSL2** — `RuntimeError: UVA is not available`,
    which names neither WSL2 nor the cause. vLLM disables pinned host memory on
    WSL2 by default and its UVA probe is just `is_pin_memory_available()`.
    `VLLM_WSL2_ENABLE_PIN_MEMORY=1` fixes it.
13. **The first benchmark rate-limited itself**, reporting 273/288 failures with
    healthy latencies. They were 429s from our own gateway at its 60 rpm
    default. The harness now counts them separately and refuses to mark such a
    run valid.

## Next steps

1. Bring up `docker compose` on a CUDA host and run `make bench` to fill in
   `docs/results.md` §4.
2. Flip the repository to public and add the URL to the CV bullet.
