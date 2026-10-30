#!/usr/bin/env bash
# End-to-end check of the CPU serving profile: vLLM's CPU backend behind the
# authenticated gateway, brought up with the real compose files.
#
#   MERGED_MODEL_DIR=path/to/hf-model-dir bash scripts/cpu_serving_smoke.sh
#
# MERGED_MODEL_DIR defaults to artifacts/merged/current, the fine-tuned model.
# CI points it at a small public stand-in model instead (see ci.yml), because
# the fine-tuned weights are not published; what CI proves is the serving
# path, not the model's answers.
#
# Checks, through the gateway only (vLLM is never published):
#   * /healthz and /readyz, and /v1/models lists the served name
#   * 401 with no key and with a wrong key
#   * 200 with a valid key, and the response carries token usage
#   * 429 with Retry-After once a low-limit key is over its request budget
#   * per-key token counters appear in /metrics, streaming included
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

export MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-./artifacts/merged/current}"
# Colon-separated on every OS; Compose would otherwise expect ";" on Windows.
export COMPOSE_PATH_SEPARATOR=":"
export COMPOSE_FILE="docker-compose.yml:docker-compose.cpu.yml"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-dllm-cpu-smoke}"
export GATEWAY_PORT="${GATEWAY_PORT:-18080}"
BASE="http://localhost:${GATEWAY_PORT}"
MODEL="${SERVED_MODEL_NAME:-domain-sql}"

if [ ! -f "$MERGED_MODEL_DIR/config.json" ]; then
  echo "no model at $MERGED_MODEL_DIR (expected a Hugging Face model directory)" >&2
  exit 2
fi

# Two keys, minted the same way `dllm serve keys` does (SHA-256 of the
# plaintext), without needing the package installed on the host. Key B is
# limited to 6 requests a minute, which with the default burst of 10 at 60 rpm
# gives it a bucket one request deep.
KEYS_JSON=$("$PY" -c '
import hashlib, json, secrets
out = []
for name, rpm in (("a", None), ("b", 6)):
    raw = "dllm_sk_" + secrets.token_urlsafe(32)
    rec = {"key_id": "smoke-" + name + "-" + secrets.token_hex(4),
           "key_hash": hashlib.sha256(raw.encode()).hexdigest(), "tier": "default"}
    if rpm:
        rec["rpm"] = rpm
    out.append({"raw": raw, "rec": rec})
print(json.dumps(out))
')
field() { "$PY" -c "import json,sys; d=json.loads(sys.argv[1]); print($1)" "$KEYS_JSON"; }
KEY_A=$(field 'd[0]["raw"]')
KEY_A_ID=$(field 'd[0]["rec"]["key_id"]')
KEY_B=$(field 'd[1]["raw"]')
export DLLM_GATEWAY_KEYS
DLLM_GATEWAY_KEYS=$(field 'json.dumps([e["rec"] for e in d])')

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "--- compose ps ---"; docker compose ps -a || true
    echo "--- vllm logs (tail) ---"; docker compose logs --no-color --tail 80 vllm || true
    echo "--- gateway logs (tail) ---"; docker compose logs --no-color --tail 40 gateway || true
  fi
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "=== bring up the CPU profile (model: $MERGED_MODEL_DIR) ==="
started=$(date +%s)
docker compose up -d --build --wait --wait-timeout "${WAIT_TIMEOUT:-900}"
echo "stack healthy after $(( $(date +%s) - started )) s"

code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
chat() {
  curl -s -D "$1" -o "$2" -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' -H "Authorization: Bearer $3" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write SQL to count the rows in table t.\"}],\"max_tokens\":16,\"temperature\":0}"
}
TMP=$(mktemp -d)

echo "=== health ==="
[ "$(code "$BASE/healthz")" = 200 ] || fail "/healthz"
[ "$(code "$BASE/readyz")" = 200 ] || fail "/readyz: gateway cannot reach vLLM"
curl -s "$BASE/v1/models" -H "Authorization: Bearer $KEY_A" | grep -q "\"$MODEL\"" \
  || fail "/v1/models does not list $MODEL"

echo "=== auth ==="
c=$(code -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{}')
[ "$c" = 401 ] || fail "no key returned $c, expected 401"
c=$(code -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dllm_sk_wrong' -d '{}')
[ "$c" = 401 ] || fail "wrong key returned $c, expected 401"

echo "=== completion through the gateway ==="
c=$(chat "$TMP/h1" "$TMP/b1" "$KEY_A")
[ "$c" = 200 ] || { cat "$TMP/b1"; fail "valid key returned $c"; }
"$PY" - "$TMP/b1" <<'PYEOF'
import json, sys
body = json.load(open(sys.argv[1]))
text = body["choices"][0]["message"]["content"]
usage = body["usage"]
assert usage["completion_tokens"] > 0 and usage["prompt_tokens"] > 0, usage
print(f"completion ({usage['completion_tokens']} tokens): {text!r}")
PYEOF

echo "=== streaming, with usage ==="
curl -sf -N -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY_A" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
  > "$TMP/stream"
grep -q '^data: \[DONE\]' "$TMP/stream" || fail "stream did not finish with [DONE]"

echo "=== rate limit ==="
c=$(chat "$TMP/h2" "$TMP/b2" "$KEY_B")
[ "$c" = 200 ] || fail "low-limit key's first request returned $c"
c=$(chat "$TMP/h3" "$TMP/b3" "$KEY_B")
[ "$c" = 429 ] || fail "low-limit key's second request returned $c, expected 429"
grep -qi '^retry-after: [0-9]' "$TMP/h3" || fail "429 without Retry-After"
echo "429 with $(grep -i '^retry-after' "$TMP/h3" | tr -d '\r')"

echo "=== per-key token counters ==="
curl -s "$BASE/metrics" > "$TMP/metrics"
"$PY" - "$TMP/metrics" "$KEY_A_ID" <<'PYEOF'
import sys
text, key_id = open(sys.argv[1]).read(), sys.argv[2]
rows = [l for l in text.splitlines()
        if l.startswith("dllm_gateway_tokens_total{") and f'key_id="{key_id}"' in l]
for r in rows:
    print(r)
kinds = {r.split('kind="')[1].split('"')[0]: float(r.rsplit(" ", 1)[1]) for r in rows}
assert kinds.get("prompt", 0) > 0 and kinds.get("completion", 0) > 0, kinds
PYEOF
grep -q 'dllm_gateway_rejected_total{reason="requests"} 1.0' "$TMP/metrics" \
  || fail "the 429 was not counted"

rm -rf "$TMP"
echo ""
echo "CPU serving smoke passed in $(( $(date +%s) - started )) s."
