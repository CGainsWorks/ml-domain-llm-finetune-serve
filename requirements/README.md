# Requirement sets

The dependency tree is split so that the parts of this repo which do **not**
need a GPU stay installable in seconds on a CPU-only CI runner.

| File | Installs | Used by |
|---|---|---|
| `base.txt` | config, data prep, eval scoring maths | eval harness, gate, unit tests, CI |
| `train.txt` | torch, transformers, peft, trl, mlflow | LoRA training, merge, HF eval backend |
| `serve.txt` | fastapi, uvicorn, openai client | gateway, benchmarks |
| `vllm.txt` | vLLM | inference server (Linux + CUDA only) |
| `dev.txt` | pytest, ruff | local development |

`base.txt` is a subset of every other file, so `pip install -r train.txt`
pulls it in transitively.

## Why torch is not on PyPI's default index here

The published Windows wheels for `torch` on PyPI are CPU-only. `train.txt`
therefore does not pin torch; install it explicitly for your platform first:

```bash
# Linux / Windows, CUDA 12.6
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0
```

`scripts/setup.sh` and `scripts/setup.ps1` do this for you.
