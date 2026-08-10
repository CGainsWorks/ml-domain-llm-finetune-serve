#!/usr/bin/env bash
# Bring the serving stack up and prove it works: readiness, auth, a real query,
# and a benchmark sweep.
#
# Needs a merged checkpoint and GPU passthrough into Docker. On Linux that is
# the NVIDIA container toolkit; on Windows it is Docker Desktop's WSL2 backend,
# which works -- vLLM has no Windows wheel but does not need one to run in a
# Linux container. Check with:
#
#   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
set -euo pipefail

cd "$(dirname "$0")/.."

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
MODEL_DIR="${MODEL_DIR:-artifacts/merged/current}"

# Same interpreter detection as run_smoke.sh: a Windows venv puts python in
# Scripts/, not bin/, and hardcoding bin/ makes this script Linux-only for no
# reason.
if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

step() { echo ""; echo "=== $* ==="; }

if [ ! -d "$MODEL_DIR" ]; then
  echo "No merged checkpoint at $MODEL_DIR."
  echo "Run 'make train && make merge' first, or point serve.vllm.model at a Hub id."
  exit 1
fi

step "checking GPU passthrough"
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi \
      --query-gpu=name,memory.total --format=csv,noheader; then
  echo "Docker cannot see a GPU. vLLM will not start."
  echo "Linux: install the NVIDIA container toolkit."
  echo "Windows: enable the WSL2 backend in Docker Desktop."
  exit 1
fi

step "minting an API key"
# The gateway stores only the hash. Build the store with python rather than a
# shell string: a single-element JSON array is easy to mangle into a bare
# object, and the gateway (correctly) refuses to start on one.
KEY_JSON="$($PY -m dllm.cli serve keys --count 1 | tail -1)"
DLLM_API_KEY="$(printf '%s' "$KEY_JSON" | $PY -c 'import json,sys; print(json.load(sys.stdin)["key"])')"
DLLM_GATEWAY_KEYS="$(printf '%s' "$KEY_JSON" | $PY -c '
import json, sys
e = json.load(sys.stdin)
print(json.dumps([{k: v for k, v in e.items() if k != "key"}]))')"
export DLLM_API_KEY DLLM_GATEWAY_KEYS
echo "key minted (plaintext is shown once and never stored)"

step "starting the stack"
docker compose up -d --build

step "waiting for the model to load"
# vLLM startup includes weight loading and CUDA graph capture; on a small card
# this is minutes, not seconds. readyz reports the upstream, healthz does not.
ready=""
for _ in $(seq 1 120); do
  if curl -fsS "$GATEWAY_URL/readyz" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ -z "$ready" ]; then
  echo "gateway never became ready. Last 40 lines of vllm:"
  docker compose logs --tail 40 vllm
  exit 1
fi
curl -fsS "$GATEWAY_URL/readyz"; echo

step "unauthenticated request (expect 401)"
curl -s -o /dev/null -w "  status=%{http_code}\n" -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"domain-sql","messages":[{"role":"user","content":"hi"}]}'

step "authenticated query"
cat > /tmp/dllm_demo_req.json <<'JSON'
{
  "model": "domain-sql",
  "messages": [
    {"role": "system", "content": "You are a SQL generator. Given a database schema and a question, reply with a single valid SQLite SELECT query that answers it. Output only the query. No explanation, no markdown, no trailing semicolon."},
    {"role": "user", "content": "Schema:\nCREATE TABLE players (name VARCHAR, county VARCHAR, runs INTEGER)\n\nQuestion: Who has the most runs from Yorkshire?\n\nSQL:"}
  ],
  "max_tokens": 64,
  "temperature": 0
}
JSON
curl -s -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $DLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/dllm_demo_req.json | $PY -m json.tool

step "benchmark sweep"
$PY -m dllm.cli bench run --config configs/base.yaml \
  --concurrency 1,4,16 --requests-per-level 24

echo ""
echo "Demo complete. 'docker compose down' to stop."
