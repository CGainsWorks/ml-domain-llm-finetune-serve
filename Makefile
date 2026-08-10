# Every target here is what CI runs. If it works locally it works in CI.
# Windows users: Git Bash works, or use the equivalent scripts/*.ps1.

PY      ?= python
VENV    ?= .venv
BIN     := $(VENV)/bin
CONFIG  ?= configs/base.yaml
LABEL   ?= finetuned

ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
endif

PYTHON := $(BIN)/python
DLLM   := $(PYTHON) -m dllm.cli

.DEFAULT_GOAL := help
.PHONY: help setup setup-cpu lint fmt test test-fast smoke audit data card \
        estimate baseline train merge eval gate compare promote serve-vllm \
        serve-gateway bench demo docker-build docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- environment ---------------------------------------------------------- #
setup:  ## Full install (CUDA torch + train + serve + dev)
	bash scripts/setup.sh

setup-cpu:  ## Minimal install: eval harness, gate and tests only. No torch.
	$(PY) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip -q
	$(PYTHON) -m pip install -r requirements/base.txt -r requirements/dev.txt -q
	$(PYTHON) -m pip install -e . --no-deps -q

# --- quality -------------------------------------------------------------- #
lint:  ## ruff check + format check
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

fmt:  ## Apply formatting
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

test:  ## Full test suite
	$(PYTHON) -m pytest tests -q

test-fast:  ## Skip anything marked slow or gpu
	$(PYTHON) -m pytest tests -q -m "not slow and not gpu"

# --- pipeline ------------------------------------------------------------- #
audit:  ## Licence audit. Exits non-zero on a violation. Downloads nothing.
	$(DLLM) data audit --config $(CONFIG)

data: audit  ## Prepare the dataset (audit runs first and can block it)
	$(DLLM) data prepare --config $(CONFIG)

card:  ## Regenerate docs/DATASET_CARD.md
	$(DLLM) data card --config $(CONFIG)

estimate:  ## GPU-hours and cost estimate. Run this BEFORE renting anything.
	$(DLLM) train estimate --config $(CONFIG)

baseline:  ## Evaluate the BASE model. Must run before training.
	$(DLLM) eval run --config $(CONFIG) --label baseline

train:  ## LoRA fine-tune. Re-run the same command to resume after a crash.
	$(DLLM) train run --config $(CONFIG)

merge:  ## Merge the adapter into the base weights
	# No --adapter: the CLI derives it from the config, which resolves _base_
	# includes properly. Re-parsing the YAML here did not.
	$(DLLM) model merge --config $(CONFIG)

eval:  ## Evaluate the fine-tuned model
	$(DLLM) eval run --config $(CONFIG) --label $(LABEL) --model-path artifacts/merged/current

gate:  ## Apply the promotion gate. Exits non-zero if the model must not ship.
	$(DLLM) eval gate --config $(CONFIG) --candidate evals/results/$(LABEL).json

compare:  ## Write docs/results.md
	$(DLLM) eval compare --baseline evals/results/baseline.json \
	  --candidate evals/results/$(LABEL).json

promote:  ## Register in MLflow and promote only if the gate passes
	$(DLLM) registry promote --config $(CONFIG) \
	  --candidate evals/results/$(LABEL).json --model-dir artifacts/merged/current

# --- serving -------------------------------------------------------------- #
serve-vllm:  ## Run vLLM directly (needs CUDA)
	$(DLLM) serve vllm --config $(CONFIG)

serve-gateway:  ## Run the authenticated FastAPI gateway
	$(DLLM) serve gateway --config $(CONFIG)

bench:  ## Concurrency sweep against the gateway
	$(DLLM) bench run --config $(CONFIG)

# --- one-command paths ---------------------------------------------------- #
smoke:  ## Whole pipeline end to end on CPU in a couple of minutes. What CI runs.
	bash scripts/run_smoke.sh

demo:  ## Serve the model and run the benchmark, in containers
	docker compose up --build

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

clean:  ## Remove generated artefacts. Leaves data/ and the venv alone.
	rm -rf artifacts mlruns .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
