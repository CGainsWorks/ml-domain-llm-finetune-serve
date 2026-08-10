#!/usr/bin/env bash
# Bring the serving stack up and prove it works: health, auth, a real query,
# and a short benchmark. Needs a merged checkpoint and a CUDA device.
set -euo pipefail

cd "$(dirname "$0")/.."

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"

if [ ! -d "artifacts/merged/current" ]; then
  echo "No merged checkpoint at artifacts/merged/current."
  echo "Run 'make train && make merge' first, or point serve.vllm.model at a Hub id."
  exit 1
fi

if [ -z "${DLLM_API_KEY:-}" ]; then
  echo ">> minting a demo API key"
  KEY_JSON="$(.venv/bin/python -m dllm.cli serve keys --count 1)"
  export DLLM_API_KEY="$(echo "$KEY_JSON" | python -c 'import json,sys;print(json.load(sys.stdin)["key"])')"
  export DLLM_GATEWAY_KEYS="[$(echo "$KEY_JSON" | python -c '
import json,sys
e = json.load(sys.stdin)
print(json.dumps({k: v for k, v in e.items() if k != "key"}))')]"
fi

echo ">> starting the stack"
docker compose up -d --build

echo ">> waiting for readiness"
for _ in $(seq 1 120); do
  if curl -fsS "$GATEWAY_URL/readyz" >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS "$GATEWAY_URL/readyz" | tee /dev/stderr >/dev/null

echo ""
echo ">> unauthenticated request (expect 401)"
curl -s -o /dev/null -w "  status=%{http_code}\n" -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"domain-sql","messages":[{"role":"user","content":"hi"}]}'

echo ""
echo ">> authenticated query"
curl -s -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $DLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "domain-sql",
        "messages": [
          {"role": "system", "content": "You are a SQL generator. Given a database schema and a question, reply with a single valid SQLite SELECT query that answers it. Output only the query. No explanation, no markdown, no trailing semicolon."},
          {"role": "user", "content": "Schema:\nCREATE TABLE players (name VARCHAR, county VARCHAR, runs INTEGER)\n\nQuestion: Who has the most runs from Yorkshire?\n\nSQL:"}
        ],
        "max_tokens": 64,
        "temperature": 0
      }' | python -m json.tool

echo ""
echo ">> benchmark"
.venv/bin/python -m dllm.cli bench run --config configs/base.yaml \
  --concurrency 1,4,16 --requests-per-level 24

echo ""
echo "Demo complete. 'docker compose down' to stop."
