#!/usr/bin/env bash
# Full development install: CUDA torch, training, serving and dev tooling.
#
# torch is installed first from the PyTorch index rather than PyPI, because the
# default-index wheel is CPU-only on Windows and would silently shadow a CUDA
# build installed afterwards.
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
VENV="${VENV:-.venv}"

cd "$(dirname "$0")/.."

# Git Bash on Windows ships `python` but usually not `python3`, so defaulting to
# python3 makes this script fail on exactly the machines most likely to be a
# self-hosted GPU runner.
if [ -z "${PY:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    echo "no python interpreter on PATH" >&2
    exit 1
  fi
fi
echo ">> using interpreter: $PY ($($PY --version 2>&1))"

if [ ! -d "$VENV" ]; then
  echo ">> creating virtualenv at $VENV"
  "$PY" -m venv "$VENV"
fi

if [ -d "$VENV/Scripts" ]; then
  PYBIN="$VENV/Scripts/python"
else
  PYBIN="$VENV/bin/python"
fi

echo ">> upgrading pip"
"$PYBIN" -m pip install --upgrade pip setuptools wheel -q

echo ">> installing torch $TORCH_VERSION ($CUDA_TAG)"
"$PYBIN" -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
  "torch==$TORCH_VERSION"

echo ">> installing requirements"
"$PYBIN" -m pip install -r requirements/train.txt -r requirements/serve.txt \
  -r requirements/dev.txt

echo ">> installing dllm (editable)"
"$PYBIN" -m pip install -e . --no-deps

echo ""
"$PYBIN" - <<'PYEOF'
import torch
print(f"torch {torch.__version__}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"CUDA available: {name} ({vram:.1f} GB), bf16={torch.cuda.is_bf16_supported()}")
    if vram < 10:
        print("NOTE: under 10 GB. Use configs/local_4060_8gb.yaml.")
else:
    print("CUDA NOT available. Eval and tests will run on CPU; training will not.")
PYEOF

echo ""
echo "Done. Next:"
echo "  make audit      # licence audit, downloads nothing"
echo "  make data       # prepare the dataset"
echo "  make estimate   # GPU hours, BEFORE you rent anything"
echo "  make baseline   # evaluate the base model FIRST"
