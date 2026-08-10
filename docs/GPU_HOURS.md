# GPU hours: what to rent, and for how long

**Short answer: for the reference 1.5B configuration you almost certainly do not
need to rent anything. It fits on the 8 GB RTX 4060 you already have, in roughly
3-6 hours. If you want it done in under an hour, one L40S or A100 for a single
hour covers training, both evals and the benchmarks with room to spare.**

Regenerate every number below with:

```bash
make estimate
```

---

## The measured inputs

These are not assumptions. `dllm train estimate` tokenises a sample of the real
prepared dataset with the real tokenizer:

| Quantity | Value | How |
|---|---:|---|
| Base model parameters | 1.54 B | Qwen2.5-1.5B-Instruct |
| Training examples | 20,000 | `data.max_train_examples` |
| Epochs | 2 | `train.epochs` |
| Optimiser steps | 2,500 | effective batch 16 |
| Tokens per example — mean | 121 | measured, 800-row sample |
| Tokens per example — median | 119 | measured |
| Tokens per example — p95 | 150 | measured |
| Examples truncated at length 512 | 0.0% | measured |
| **Total training tokens** | **4.82 M** | mean × 40,000 samples |
| **Total compute** | **44.6 PFLOPs** | 6·N·T with gradient checkpointing |

The token count is the thing most estimates get wrong. Assuming every example
fills `max_seq_len` would have put this at 41 M tokens and overstated the run by
**8x**. Text-to-SQL examples are short.

## The arithmetic

A forward pass is about `2·N·T` FLOPs. A full backward pass is `4·N·T` — half
for gradients with respect to activations, half with respect to weights. LoRA
freezes the base, so the weight-gradient half nearly vanishes and training costs
about `4·N·T`. Gradient checkpointing recomputes the forward during backward,
adding `2·N·T` back:

```
LoRA, no checkpointing:    4 · N · T
LoRA, with checkpointing:  6 · N · T   <- the reference config
```

Divide by the card's dense BF16 throughput times a realistic utilisation figure.

## The table (1.5B reference config)

| GPU | VRAM | Effective TFLOPs | Est. hours | $/h | Est. $ | Fits |
|---|---:|---:|---:|---:|---:|---|
| RTX 4060 8GB (yours) | 8 GB | 4.8 | **2.6** | — | — | yes (~5.3 GB) |
| RTX 4090 24GB | 24 GB | 36.3 | 0.34 | 0.40 | 0.14 | yes |
| L4 24GB | 24 GB | 14.4 | 0.86 | 0.75 | 0.64 | yes |
| A10G 24GB | 24 GB | 15.5 | 0.80 | 1.00 | 0.80 | yes |
| L40S 48GB | 48 GB | 54.3 | 0.23 | 1.40 | 0.32 | yes |
| A100 40GB | 40 GB | 99.8 | 0.12 | 1.60 | 0.20 | yes |
| H100 80GB | 80 GB | 173.2 | 0.07 | 3.00 | 0.21 | yes |

### Read the caveat before you believe the fast end of that table

The FLOPs model assumes the GPU is **compute-bound**. At an effective batch of
16 and ~121 tokens per example, each optimiser step processes only about **1,900
tokens**. That is far too few to saturate an A100, let alone an H100. Below
roughly 8k tokens per step the run is bound by per-step fixed costs — kernel
launches, the optimiser, the dataloader — not by matrix multiplies.

So:

- **The table is a lower bound.** On the fast datacentre cards expect **2-4x**
  the quoted figure. An A100 will not finish this in 7 minutes; budget 20-30.
- **The slow-card numbers are more trustworthy**, because a 4060 genuinely is
  compute-bound on this workload.
- Raising `train.per_device_batch_size` or enabling `train.packing` closes the
  gap. The reference config already uses batch 8 × accum 2 rather than 2 × 8 for
  this reason.

### Getting a real number instead of a modelled one

```bash
# ~20 optimiser steps, a couple of minutes
dllm train run -c configs/base.yaml --set train.max_steps=20

# now reads the measured samples/s and extrapolates
dllm train estimate -c configs/base.yaml
```

The estimator prints a `CALIBRATED:` line once a measured throughput exists.
Trust that over the table.

## Budget the whole session, not just training

Renting by the hour means paying for everything, including the parts that do not
touch the GPU hard.

| Stage | 1.5B on an A100 | 1.5B on the 4060 |
|---|---:|---:|
| Model download (3.1 GB) | 2-5 min | 2-5 min |
| Dataset prep (audit, clean, split, verify) | ~3 min | ~3 min |
| **Baseline eval** (500 + 200 items) | 5-10 min | ~45 min |
| **Training** | 20-30 min | 3-6 h |
| Merge + verify | 2-4 min | 3-5 min |
| Fine-tuned eval | 5-10 min | ~45 min |
| vLLM startup + benchmark sweep | 8-12 min | 10-15 min |
| **Total** | **~1 hour** | **~6-8 hours** |

**Rent one hour of an L40S or A100 and it is comfortably enough.** Rent two if
you want to try a second LoRA rank without watching the clock.

## The 7B profile

`configs/rented_a100.yaml` swaps in Qwen2.5-Coder-7B-Instruct. This one does
need rented hardware:

- ~15 GB of weights in bf16, plus activations. It does **not** fit 24 GB at a
  useful batch size, and does not fit 8 GB at all without 4-bit quantisation.
- An A100 40GB fits it at micro-batch 16 (~31 GB). An L40S 48GB or H100 80GB is
  more comfortable.
- Estimated 0.4-0.8 GPU-hours of pure training, so **budget 2 hours** for the
  full session including both evals.

Whether it is worth it is an open question: the baseline number for the 1.5B
model is the thing to look at first. If the 1.5B base already scores well, the
headroom a 7B buys may not justify 5x the serving cost.

## Before you rent: the checklist

```bash
make audit       # licence audit. Downloads nothing. Do this first.
make data        # prepare the dataset locally (CPU, ~3 min)
make estimate    # this document, regenerated with your config
make baseline    # evaluate the BASE model. Do this before renting anything.
```

Running the baseline locally before you rent is worth the wall-clock. It is the
number every later claim is measured against, it costs nothing but time on your
own card, and it means the rented hour is spent on training rather than on
establishing a before-number.

## If the run is interrupted

Re-run the identical command. `train.resume: auto` finds the newest checkpoint
and continues — optimiser state, scheduler, RNG state and dataset position
included — and reattaches to the same MLflow run. Checkpoints are written every
50 optimiser steps, so the most you can lose is a couple of minutes.

See [RUNBOOK_RENTED_GPU.md](RUNBOOK_RENTED_GPU.md) for the full session script.
