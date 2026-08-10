# Runbook: a rented GPU session

Written to be followed top to bottom on a fresh box with the meter running.
Every command is copy-pasteable. Read [GPU_HOURS.md](GPU_HOURS.md) first to
decide what to rent.

## Before you rent (on your own machine)

Do this locally. It costs nothing and it removes the two things most likely to
waste rented time: a licence problem discovered after training, and a missing
before-number.

```bash
make setup
make audit       # exits non-zero on a licence violation; downloads nothing
make data        # ~3 min on CPU
make estimate    # how long, on what, for how much
make baseline    # the base-model scorecard. Everything is measured against this.
git add evals/results/baseline.json docs/ && git commit -m "Add baseline scorecard"
```

If `make baseline` is too slow locally, run it on the rented box **first**,
before training. Do not skip it. A fine-tuned number with no baseline is not a
result.

---

## On the rented box

### 1. Sanity check the hardware (30 seconds)

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

If `is_bf16_supported()` is False you are on pre-Ampere hardware: set
`--set train.bf16=false --set train.fp16=true`.

### 2. Install

```bash
git clone https://github.com/CGainsWorks/ml-domain-llm-finetune-serve.git
cd ml-domain-llm-finetune-serve
make setup                       # CUDA_TAG=cu126 by default
```

### 3. Prepare data and confirm the estimate

```bash
make audit
make data
make estimate
```

### 4. Calibrate before committing (2 minutes, saves hours)

```bash
dllm train run -c configs/base.yaml --set train.max_steps=20 \
                                    --set train.run_name=calib
dllm train estimate -c configs/base.yaml --set train.run_name=calib
```

Look for the `CALIBRATED:` line. If it says four hours and you rented one, stop
and change something now — smaller `data.max_train_examples`, fewer epochs, or a
bigger card — rather than discovering it at minute 55.

Also check `artifacts/checkpoints/calib/heartbeat.json` for peak VRAM. If it is
above about 90% of the card, lower `train.per_device_batch_size`.

### 5. Baseline, if you have not already

```bash
make baseline
```

### 6. Train

```bash
# Run under tmux. An SSH drop must not kill the run.
tmux new -s train
make train
# detach: Ctrl-B then D
```

Monitor from another shell without attaching:

```bash
watch -n 30 cat artifacts/checkpoints/sql-lora-1p5b/heartbeat.json
```

That file carries `global_step`, `max_steps`, `steps_per_s`, `eta_s` and peak
VRAM, and is written atomically, so it is always valid to read.

**If it dies, re-run the identical command.** `train.resume: auto` picks up the
newest checkpoint and reattaches to the same MLflow run. Checkpoints land every
50 steps.

### 7. Merge

```bash
make merge
```

This verifies that the merged checkpoint reproduces the adapter-on-base outputs
before writing it, and refuses to export if it does not.

### 8. Evaluate the fine-tune and compare

```bash
make eval        # writes evals/results/finetuned.json
make compare     # writes docs/results.md
make gate        # exit 0 = promotable, exit 1 = do not ship
```

### 9. Register, gated

```bash
make promote
```

Registers the version with its scorecard and gate report attached. The
`deployment_target` alias moves **only** if the gate passed; a blocked candidate
is still registered, tagged `blocked`, so the record survives.

### 10. Serve and benchmark

```bash
dllm serve keys --count 1        # save the plaintext key; it is shown once
export DLLM_API_KEY=dllm_sk_...
export DLLM_GATEWAY_KEYS='[{"key_id":"...","key_hash":"...","tier":"default"}]'

docker compose up -d --build
curl -fsS localhost:8080/readyz

make bench                       # concurrency sweep -> artifacts/bench/latest.json
```

### 11. Get your artefacts off the box before you stop paying

The rented volume disappears with the instance. In order of importance:

```bash
# 1. Scorecards, gate report, benchmarks -- small, and they are the evidence
git add evals/results/ docs/results.md docs/gate_report.md
git commit -m "Add fine-tuned scorecard and benchmarks"
git push

# 2. The adapter -- tens of MB, the actual output of the run
tar czf adapter.tar.gz artifacts/checkpoints/sql-lora-1p5b/adapter
#    upload to a GitHub Release, or push to the Hub:
#    huggingface-cli upload <you>/domain-sql-lora artifacts/checkpoints/sql-lora-1p5b/adapter

# 3. MLflow history -- optional but cheap
tar czf mlruns.tar.gz mlruns/
```

Do **not** commit the merged checkpoint. It is several GB, it is regenerable
from the adapter in two minutes, and `.gitattributes` routes weights to LFS
precisely so a stray `git add .` does not put it in the history.

---

## Things that waste rented time, in order of how often they happen

| Failure | Cost | Prevention |
|---|---|---|
| No baseline, so the result means nothing | the whole session | `make baseline` first; the gate refuses without one |
| Typo in a config key, discovered at step 900 | 30+ min | configs are strict; `make estimate` validates them |
| OOM at step 900 | 30+ min | the 20-step calibration reports peak VRAM |
| SSH drop kills the run | up to the full run | tmux, plus `resume: auto` |
| Disk fills with checkpoints | the run | `save_total_limit: 3` |
| Instance stopped before artefacts were copied off | everything | step 11 |
| Downloading a 15 GB model on the clock | 10-20 min | pre-warm `HF_HOME`, or pick the 1.5B |

## Cost control

- Stop the instance the moment step 11 finishes. Storage-only billing is much
  cheaper than a running GPU.
- Spot and preemptible instances are fine here **because the run is resumable**.
  That is what `save_steps: 50` and `resume: auto` are for.
- Do not rent an H100 for a 1.5B LoRA. You pay for memory bandwidth this
  workload cannot use — see the caveat in [GPU_HOURS.md](GPU_HOURS.md).
