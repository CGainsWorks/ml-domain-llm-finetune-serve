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

# Finding a usable interpreter is genuinely fiddly on Windows, and getting it
# wrong builds a venv that fails much later with a confusing error:
#
#   * `command -v python3` succeeds even when python3 is the Microsoft Store
#     stub in WindowsApps. Running it prints an advert and exits non-zero.
#   * `python` is often an older system install. This project needs >= 3.10
#     (transformers 5, peft, trl, vLLM and mlflow all require it), and a 3.9
#     venv gets most of the way through pip before failing.
#
# So: try candidates in order, and accept one only if it actually executes and
# reports a high enough version.
MIN_MINOR=10

usable_python() {
  local cand="$1" out minor
  command -v "$cand" >/dev/null 2>&1 || return 1
  out=$("$cand" -c 'import sys; print(sys.version_info[0], sys.version_info[1])' 2>/dev/null) || return 1
  set -- $out
  [ "${1:-0}" = "3" ] || return 1
  minor="${2:-0}"
  [ "$minor" -ge "$MIN_MINOR" ] || return 1
  return 0
}

if [ -n "${PY:-}" ]; then
  usable_python "$PY" || {
    echo "PY=$PY is not a usable Python >= 3.$MIN_MINOR" >&2
    exit 1
  }
else
  PY=""
  for cand in python3.12 python3.11 python3.10 python3 python; do
    if usable_python "$cand"; then PY="$cand"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "no Python >= 3.$MIN_MINOR found on PATH." >&2
    echo "Tried: python3.12 python3.11 python3.10 python3 python" >&2
    echo "Checked versions:" >&2
    for cand in python3 python; do
      command -v "$cand" >/dev/null 2>&1 && \
        echo "  $cand -> $("$cand" --version 2>&1 | head -1)" >&2
    done
    echo "Set PY=/path/to/python3.12 explicitly, or install a newer Python." >&2
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
