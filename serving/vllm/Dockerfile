# vLLM inference server, OpenAI-compatible.
#
# Uses the official vLLM image rather than building CUDA from scratch: the
# wheel is matched to a specific CUDA and driver combination and rebuilding it
# is a reliable way to lose an afternoon.
#
# The same file builds the CPU profile. docker-compose.cpu.yml passes
#   VLLM_IMAGE=vllm/vllm-openai-cpu:v0.26.0
# which is vLLM's own CPU backend at the same version, so both profiles run
# one engine and one command line.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.26.0
FROM ${VLLM_IMAGE}

# The merged checkpoint is mounted, not baked in. A multi-GB layer per model
# version makes the image unusable in CI and unpushable over most connections.
ENV MODEL_DIR=/models/current \
    SERVED_MODEL_NAME=domain-sql \
    MAX_MODEL_LEN=2048 \
    GPU_MEMORY_UTILIZATION=0.85 \
    MAX_NUM_SEQS=64 \
    DTYPE=bfloat16 \
    EXTRA_ARGS=""

EXPOSE 8000

# /health, plus an optional one-time warm-up completion (WARMUP_ON_HEALTHCHECK=1,
# used by the CPU profile). See the script for why.
COPY --chmod=0755 serving/vllm/healthcheck.sh /usr/local/bin/dllm-vllm-health

HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=10 \
  CMD dllm-vllm-health || exit 1

# Override the base image's entrypoint so the env vars above are expanded.
ENTRYPOINT []
CMD ["/bin/sh", "-c", "\
  exec python3 -m vllm.entrypoints.openai.api_server \
    --model \"$MODEL_DIR\" \
    --served-model-name \"$SERVED_MODEL_NAME\" \
    --host 0.0.0.0 --port 8000 \
    --dtype \"$DTYPE\" \
    --max-model-len \"$MAX_MODEL_LEN\" \
    --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\" \
    --max-num-seqs \"$MAX_NUM_SEQS\" \
    --enable-prefix-caching \
    $EXTRA_ARGS"]
