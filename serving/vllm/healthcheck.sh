#!/bin/sh
# Container health for the vLLM server.
#
# /health only says the HTTP server is up. On the CPU backend the first real
# forward pass after start-up is much slower than the rest (80 s for the 1.5B
# model on an AVX2 desktop, against about 4 s for the next request), so a
# container that reports healthy on /health alone tells the gateway, and
# `docker compose up --wait`, that the model is ready when it is not.
#
# With WARMUP_ON_HEALTHCHECK=1 (set by docker-compose.cpu.yml) the first
# passing check also runs one 1-token completion and records that it did, so
# "healthy" means "has answered a request". Later checks are /health only.
set -e

curl -fsS http://localhost:8000/health >/dev/null

[ "${WARMUP_ON_HEALTHCHECK:-0}" = "1" ] || exit 0
[ -f /tmp/dllm-warm ] && exit 0

curl -fsS -m 900 http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"SELECT\",\"max_tokens\":1}" >/dev/null
touch /tmp/dllm-warm
