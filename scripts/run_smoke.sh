#!/usr/bin/env bash
# Whole pipeline end to end, CPU only, in a couple of minutes.
#
# This is what CI runs on every push. It proves the stages are wired together:
# audit -> prepare -> baseline eval -> train -> merge -> eval -> gate.
#
# It uses a random-weight tiny model. The numbers it produces are meaningless
# and it deliberately does not gate on them -- see configs/smoke.yaml.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d ".venv/Scripts" ]; then
  DLLM=".venv/Scripts/python -m dllm.cli"
elif [ -d ".venv/bin" ]; then
  DLLM=".venv/bin/python -m dllm.cli"
else
  DLLM="python -m dllm.cli"
fi

CONFIG="configs/smoke.yaml"

# Isolate the smoke run's data. Without this a 64-example smoke prepare would
# overwrite the real prepared splits and quietly invalidate every scorecard on
# the machine. Task definitions resolve a leading "data/" against this.
export DLLM_DATA_DIR="${DLLM_DATA_DIR:-$PWD/artifacts/smoke_data}"
mkdir -p "$DLLM_DATA_DIR"
echo "smoke data directory: $DLLM_DATA_DIR"

# And its generated docs. The smoke run regenerates the dataset card and the
# licence audit from a 64-example toy dataset; writing those into the real
# docs/ replaced the committed cards with smoke numbers, which is exactly the
# kind of change that gets committed by accident.
export DLLM_DOCS_DIR="${DLLM_DOCS_DIR:-$PWD/artifacts/smoke_docs}"
mkdir -p "$DLLM_DOCS_DIR"
echo "smoke docs directory: $DLLM_DOCS_DIR"

step() { echo ""; echo "=== $* ==="; }

step "1/7 licence audit"
$DLLM data audit --config "$CONFIG"

step "2/7 dataset preparation"
$DLLM data prepare --config "$CONFIG"

step "3/7 dataset card"
$DLLM data card --config "$CONFIG"

step "4/7 baseline eval (before any training)"
$DLLM eval run --config "$CONFIG" --label smoke-baseline

step "5/7 LoRA training (8 steps)"
$DLLM train run --config "$CONFIG"

step "6/7 merge adapter"
$DLLM model merge --config "$CONFIG" \
  --adapter artifacts/checkpoints/smoke/adapter \
  --out artifacts/merged/smoke

step "7/7 eval the merged model and compare"
$DLLM eval run --config "$CONFIG" --label smoke-finetuned \
  --model-path artifacts/merged/smoke
$DLLM eval compare \
  --baseline evals/results/smoke-baseline.json \
  --candidate evals/results/smoke-finetuned.json \
  --out artifacts/smoke_results.md

echo ""
echo "=== gate (expected to BLOCK: a random-weight model must not pass) ==="
if $DLLM eval gate --config configs/base.yaml \
     --candidate evals/results/smoke-finetuned.json \
     --baseline evals/results/smoke-baseline.json \
     --report artifacts/smoke_gate.md; then
  echo "SMOKE FAILED: the gate approved a random-weight model."
  exit 1
else
  echo "OK: the gate blocked it, which is the point."
fi

echo ""
echo "Smoke test passed. Every stage ran and the gate behaved correctly."
