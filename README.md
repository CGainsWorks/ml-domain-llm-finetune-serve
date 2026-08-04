# ml-domain-llm-finetune-serve

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

The full lifecycle of a domain-adapted language model: dataset preparation with a licence audit, LoRA fine-tuning, MLflow experiment tracking and a model registry, an automated evaluation harness gating promotion, and high-throughput serving via vLLM behind FastAPI.

---

## What this covers

CV entries this repository is the evidence for:

- Fine-Tuned Domain LLM, training and serving

---

## Requirements

### Hardware

- GPU with sufficient VRAM for LoRA fine-tuning, local or rented
- Inference GPU for vLLM serving

### Software

- PyTorch
- Hugging Face Transformers and PEFT
- LoRA
- MLflow tracking and registry
- Automated eval harness (lm-eval-harness or custom)
- vLLM
- FastAPI
- Docker
- CI/CD

### Report and documentation deliverables

- Dataset card and licence audit
- Training config and hyperparameters
- Base against fine-tuned eval comparison
- Throughput and latency benchmarks
- Model card with limitations

---

## Inputs

- Base model checkpoint
- Domain instruction dataset
- Eval set
- Training config
- Serving config

## Outputs

- LoRA adapter weights
- Merged checkpoint
- MLflow runs and a registered model version
- Eval scorecard
- OpenAI-compatible inference endpoint
- Container image

---

## Intended structure

```
configs/                model, lora, train, eval, serve
src/
  data/                   dataset prep, formatting, splits
  train/                  LoRA training loop, MLflow logging
  merge/                  adapter merging
  eval/                   harness, task definitions, scoring
  serve/                  vLLM launcher, FastAPI wrapper
evals/                    task definitions and results
serving/vllm/             serving config
.github/workflows/        train, eval-gate, deploy
docs/                     MODEL_CARD.md, DATASET_CARD.md, results.md
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] Domain dataset prepared with a licence audit and dataset card
- [ ] Baseline evaluation of the base model
- [ ] LoRA fine-tuning run with MLflow tracking
- [ ] Adapter merge and checkpoint export
- [ ] Automated eval harness with defined tasks
- [ ] Base vs fine-tuned scorecard
- [ ] MLflow model registry with promotion gates
- [ ] vLLM serving with an OpenAI-compatible API
- [ ] FastAPI wrapper with auth and rate limiting
- [ ] Throughput and latency benchmarks
- [ ] CI/CD pipeline gating deployment on eval results
- [ ] Model card

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: ml-domain-llm-finetune-serve
The full lifecycle of a domain-adapted language model: dataset preparation with a licence audit, LoRA fine-tuning, MLflow experiment tracking and a model registry, an automated evaluation harness gating promotion, and high-throughput serving via vLLM behind FastAPI.

TARGET STACK
- PyTorch
- Hugging Face Transformers and PEFT
- LoRA
- MLflow tracking and registry
- Automated eval harness (lm-eval-harness or custom)
- vLLM
- FastAPI
- Docker
- CI/CD

HARDWARE CONTEXT
- GPU with sufficient VRAM for LoRA fine-tuning, local or rented
- Inference GPU for vLLM serving

THE SYSTEM MUST TAKE THESE INPUTS
- Base model checkpoint
- Domain instruction dataset
- Eval set
- Training config
- Serving config

AND PRODUCE THESE OUTPUTS
- LoRA adapter weights
- Merged checkpoint
- MLflow runs and a registered model version
- Eval scorecard
- OpenAI-compatible inference endpoint
- Container image

INTENDED REPOSITORY LAYOUT
configs/                model, lora, train, eval, serve
src/
  data/                   dataset prep, formatting, splits
  train/                  LoRA training loop, MLflow logging
  merge/                  adapter merging
  eval/                   harness, task definitions, scoring
  serve/                  vLLM launcher, FastAPI wrapper
evals/                    task definitions and results
serving/vllm/             serving config
.github/workflows/        train, eval-gate, deploy
docs/                     MODEL_CARD.md, DATASET_CARD.md, results.md

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.


START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
