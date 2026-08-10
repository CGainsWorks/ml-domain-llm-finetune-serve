# Status

Last updated: 2026-08-10

## What is done and verified

| Component | State | How it was verified |
|---|---|---|
| Licence audit | Working | Passes on the default mix; **blocks** on the deliberately-registered non-commercial source. Exits non-zero. |
| Dataset preparation | Working | 78,577 raw rows → 20,000 train / 500 validation / 500 test / 200 guard. 193 unparseable and 86 non-executable gold queries dropped. |
| Eval harness | Working | 238 unit tests. Executed against the real base model twice. |
| **Baseline scorecard** | **Complete** | `evals/results/baseline.json` — 500 + 200 items, 1,690 s on an RTX 4060. |
| LoRA training | Working | Run end to end on CPU with a tiny model: adapter written, MLflow run created, checkpoints saved. |
| Resume from checkpoint | Working | Killed and re-run; resumed from `checkpoint-8` and reattached to the **same** MLflow run id. |
| Adapter merge | Working | Merged and verified against adapter-on-base outputs before export. |
| Promotion gate | Working | Blocks a candidate equal to the baseline (delta 0 < required 0.05), exit code 1, with a report naming the failing check. |
| MLflow registry | Implemented | SQLite backend. Not yet exercised against a real trained model. |
| vLLM launcher | Implemented | Command construction unit-tested. **Not run against a GPU.** |
| FastAPI gateway | Working | 17 HTTP-level tests: auth fails closed, rate limits fire in the request path, allowlist and size cap enforced. |
| Benchmarks | Implemented | Harness complete. **No numbers yet** — needs a served model. |
| CI/CD | Implemented | Workflows written. **Not yet run on GitHub Actions.** |
| Docs | Complete | Model card, dataset card, licence audit, architecture, GPU hours, runbook. |

## What is not done

- **No fine-tuning run has been executed.** No fine-tuned numbers are claimed
  anywhere in this repository. `docs/results.md` says `NOT YET RUN` where the
  numbers would go.
- **The serving stack has never run on a GPU.** vLLM has no Windows wheel, and
  this was built on Windows. The gateway is tested against a mocked upstream;
  the vLLM half is unverified in practice.
- **No benchmark numbers.** The harness exists; it needs a served model.
- **CI has not run.** The workflows are written but this repository has had no
  push to GitHub Actions yet.

## Next steps

1. `make train` on the 4060 (~2.6-6 h) or on a rented card (~1 h all-in).
2. `make merge && make eval && make compare && make gate`.
3. Bring up `docker compose` on a CUDA host and run `make bench`.
4. Fill in `docs/results.md` sections 3 and 4 from the generated output.
5. Flip the repository to public and add the URL to the CV bullet.

## Notable findings so far

**A measurement artefact worth 15.7 accuracy points.** The eval harness
originally built its verification databases with SQLite's default
case-sensitive, strictly-typed comparison, against a dataset written in MySQL.
That scored `WHERE role = 'Taylor'` as wrong against a gold of
`WHERE role = "taylor"`. Correcting the collation and affinity moved the base
model from 0.6296 to 0.7865. Left unfixed, fine-tuning would have recovered
most of those points by learning the dataset's lowercasing convention, and it
would have been reported as improved SQL ability. Written up in
`docs/results.md` §2.

**Text matching under-reports this task by 19 points.** Base model
`exact_match_norm` is 0.596 against `exec_acc` 0.786.

**The training workload is much smaller than it looks.** Measured mean is 121
tokens per example, not the 1024 `max_seq_len` would imply — an 8× difference in
the compute estimate. Training length and model length are now separate config
fields for this reason.
