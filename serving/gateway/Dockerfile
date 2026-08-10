# FastAPI gateway: auth, rate limiting and request accounting in front of vLLM.
#
# CPU-only and deliberately small. It must be cheap to run several replicas of
# this next to one expensive GPU container.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DLLM_REPO_ROOT=/app

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Requirements first so the dependency layer caches across source edits.
COPY requirements/ requirements/
RUN pip install -r requirements/serve.txt

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY evals/ evals/
RUN pip install -e . --no-deps

# Do not run as root.
RUN useradd --create-home --uid 10001 gateway && chown -R gateway:gateway /app
USER gateway

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8080/healthz || exit 1

CMD ["python", "-m", "dllm.cli", "serve", "gateway", "--config", "configs/base.yaml"]
