"""``dllm`` command line.

Every stage of the pipeline is reachable from here, and every stage takes the
same ``--config`` plus ``--set key=value`` overrides, so a rented-GPU session
does not turn into a hunt for the right script.

Commands that make a pass/fail judgement (``data audit``, ``eval gate``) exit
non-zero on failure. That is what CI blocks on.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path

import typer

from dllm.config import load_config
from dllm.paths import DOCS, ROOT

app = typer.Typer(add_completion=False, help="Domain LLM: prepare, train, evaluate, serve.")
data_app = typer.Typer(help="Dataset preparation and licence auditing.")
eval_app = typer.Typer(help="Evaluation harness, scorecards and the promotion gate.")
train_app = typer.Typer(help="LoRA fine-tuning.")
model_app = typer.Typer(help="Adapter merging and export.")
registry_app = typer.Typer(help="MLflow model registry.")
serve_app = typer.Typer(help="Serving.")
bench_app = typer.Typer(help="Throughput and latency benchmarks.")

app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")
app.add_typer(model_app, name="model")
app.add_typer(registry_app, name="registry")
app.add_typer(serve_app, name="serve")
app.add_typer(bench_app, name="bench")

ConfigOpt = typer.Option("configs/base.yaml", "--config", "-c", help="Pipeline config YAML.")
SetOpt = typer.Option(None, "--set", "-s", help="Override, e.g. --set train.epochs=1")


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 so reports print on a Windows console.

    Scorecards and gate reports contain Δ, ▲, ▼, ✅ and ❌. On Windows the
    console encoding is cp1252 by default, and printing any of them raises
    UnicodeEncodeError -- which killed the end-to-end smoke run at the final
    step, after every stage had already succeeded. The markdown files are
    written as UTF-8 regardless; this only affects the terminal.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # An already-detached or non-reconfigurable stream is not fatal.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def _setup_logging(verbose: bool = False) -> None:
    _force_utf8_output()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@data_app.command("audit")
def data_audit(config: str = ConfigOpt, set_: list[str] = SetOpt) -> None:
    """Audit dataset licences. Exits non-zero on a violation. Downloads nothing."""
    from dllm.data.licence_audit import audit, write_audit

    cfg = load_config(config, set_)
    manifest = audit(
        [cfg.data.source, "dolly_guard"],
        require_commercial=cfg.data.require_commercial_licence,
        base_model_licence=cfg.model.licence,
        base_model_commercial=cfg.model.commercial_use,
    )
    write_audit(manifest, ROOT / "data" / "licence_audit.json", DOCS / "LICENCE_AUDIT.md")

    for f in manifest["findings"]:
        typer.echo(f"[{f['level'].upper():5s}] {f['source']}: {f['message']}")
    typer.echo(
        f"\nAudit {'PASSED' if manifest['passed'] else 'FAILED'} "
        f"({manifest['error_count']} error, {manifest['warn_count']} warning)"
    )
    typer.echo(f"Report: {DOCS / 'LICENCE_AUDIT.md'}")
    raise typer.Exit(0 if manifest["passed"] else 1)


@data_app.command("prepare")
def data_prepare(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    skip_guard: bool = typer.Option(False, help="Skip building the forgetting-guard set."),
    skip_gold_check: bool = typer.Option(
        False, help="Skip verifying that eval gold queries execute (faster, less clean)."
    ),
) -> None:
    """Audit, download, clean, split and write the dataset."""
    from dllm.data.prepare import prepare

    cfg = load_config(config, set_)
    summary = prepare(
        cfg.data,
        cfg.model,
        include_guard=not skip_guard,
        verify_eval_gold=not skip_gold_check,
    )
    typer.echo(json.dumps(summary, indent=2))


@data_app.command("card")
def data_card(config: str = ConfigOpt, set_: list[str] = SetOpt) -> None:
    """Regenerate docs/DATASET_CARD.md from the registry and prep statistics."""
    from dllm.data.dataset_card import write_dataset_card

    cfg = load_config(config, set_)
    path = write_dataset_card(cfg)
    typer.echo(f"wrote {path}")


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@eval_app.command("run")
def eval_run(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    label: str = typer.Option(..., "--label", "-l", help="Scorecard label, e.g. 'baseline'."),
    model_path: str = typer.Option(None, help="Local checkpoint to evaluate instead of the base."),
    adapter: str = typer.Option(None, help="LoRA adapter to attach to the base model."),
) -> None:
    """Evaluate a model and write a scorecard."""
    from dllm.eval.runner import evaluate
    from dllm.eval.scorecard import render_single

    cfg = load_config(config, set_)
    card, path = evaluate(
        cfg.eval,
        cfg.model,
        label=label,
        model_path=model_path,
        adapter_path=adapter,
        quant=cfg.quant,
    )
    typer.echo(render_single(card))
    typer.echo(f"scorecard: {path}")


@eval_app.command("rescore")
def eval_rescore(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    scorecard: str = typer.Option(..., help="Existing scorecard whose records to replay."),
    label: str = typer.Option(..., "--label", "-l", help="Label for the new scorecard."),
) -> None:
    """Re-score saved predictions with the current scorer. No GPU, no generation."""
    from dllm.eval.runner import rescore
    from dllm.eval.scorecard import render_single

    cfg = load_config(config, set_)
    card, path = rescore(scorecard, cfg.eval, label=label)
    typer.echo(render_single(card))
    typer.echo(f"scorecard: {path}")


@eval_app.command("gate")
def eval_gate(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    candidate: str = typer.Option(..., help="Candidate scorecard JSON."),
    baseline: str = typer.Option(None, help="Baseline scorecard JSON (defaults to the config)."),
    report: str = typer.Option("docs/gate_report.md", help="Where to write the report."),
) -> None:
    """Decide whether a model may be promoted. Exits 1 if it may not."""
    from dllm.eval.gate import decide
    from dllm.eval.scorecard import Scorecard, render_comparison

    cfg = load_config(config, set_)
    cand = Scorecard.load(candidate)
    base_path = baseline or cfg.eval.baseline_scorecard
    base = Scorecard.load(base_path) if Path(base_path).is_file() else None

    decision = decide(cand, cfg.eval.thresholds, base)

    body = decision.render()
    if base is not None:
        body += "\n" + render_comparison(base, cand)

    out = ROOT / report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    typer.echo(body)
    typer.echo(f"report: {out}")
    raise typer.Exit(0 if decision.promote else 1)


@eval_app.command("compare")
def eval_compare(
    baseline: str = typer.Option(..., help="Baseline scorecard JSON."),
    candidate: str = typer.Option(..., help="Candidate scorecard JSON."),
    out: str = typer.Option("docs/results.md", help="Markdown output path."),
) -> None:
    """Write the base vs fine-tuned scorecard comparison."""
    from dllm.eval.scorecard import Scorecard, render_comparison

    base = Scorecard.load(baseline)
    cand = Scorecard.load(candidate)
    body = render_comparison(base, cand)
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    typer.echo(body)
    typer.echo(f"wrote {path}")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
@train_app.command("run")
def train_run(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    resume: str = typer.Option(
        None, help="Checkpoint directory to resume from. Default: auto-detect the latest."
    ),
) -> None:
    """Run LoRA fine-tuning with MLflow tracking. Resumable."""
    from dllm.train.lora import train

    cfg = load_config(config, set_)
    result = train(cfg, resume_from=resume)
    typer.echo(json.dumps(result, indent=2, default=str))


@train_app.command("estimate")
def train_estimate(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    gpu: str = typer.Option(None, help="Only show this GPU, e.g. 'A100-40GB'."),
) -> None:
    """Estimate GPU hours and rental cost before starting a run."""
    from dllm.train.estimate import render_estimate

    cfg = load_config(config, set_)
    typer.echo(render_estimate(cfg, only_gpu=gpu))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@model_app.command("merge")
def model_merge(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    adapter: str = typer.Option(..., help="LoRA adapter directory."),
    out: str = typer.Option("artifacts/merged/current", help="Output directory."),
    verify: bool = typer.Option(True, help="Check merged outputs match adapter-on-base."),
) -> None:
    """Merge a LoRA adapter into the base weights and export a checkpoint."""
    from dllm.merge.merge_adapter import merge_and_export

    cfg = load_config(config, set_)
    info = merge_and_export(cfg, adapter_path=adapter, out_dir=out, verify=verify)
    typer.echo(json.dumps(info, indent=2, default=str))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
@registry_app.command("promote")
def registry_promote(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    candidate: str = typer.Option(..., help="Candidate scorecard JSON."),
    baseline: str = typer.Option(None, help="Baseline scorecard JSON."),
    model_dir: str = typer.Option(None, help="Merged model directory to register."),
    name: str = typer.Option("domain-sql", help="Registered model name."),
    stage: str = typer.Option("Staging", help="Alias to assign on success."),
) -> None:
    """Register a model version and promote it only if the eval gate passes."""
    from dllm.registry.promote import promote

    cfg = load_config(config, set_)
    result = promote(
        cfg,
        candidate_path=candidate,
        baseline_path=baseline,
        model_dir=model_dir,
        name=name,
        alias=stage,
    )
    typer.echo(json.dumps(result, indent=2, default=str))
    raise typer.Exit(0 if result["promoted"] else 1)


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #
@serve_app.command("vllm")
def serve_vllm(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    print_only: bool = typer.Option(False, help="Print the command instead of running it."),
) -> None:
    """Launch the vLLM OpenAI-compatible server."""
    from dllm.serve.vllm_launcher import build_command, launch

    cfg = load_config(config, set_)
    cmd = build_command(cfg.serve.vllm)
    if print_only:
        typer.echo(" ".join(cmd))
        return
    raise typer.Exit(launch(cfg.serve.vllm))


@serve_app.command("gateway")
def serve_gateway(config: str = ConfigOpt, set_: list[str] = SetOpt) -> None:
    """Run the FastAPI gateway (auth + rate limiting) in front of vLLM."""
    import uvicorn

    from dllm.serve.gateway.app import create_app

    cfg = load_config(config, set_)
    uvicorn.run(
        create_app(cfg.serve.gateway),
        host=cfg.serve.gateway.host,
        port=cfg.serve.gateway.port,
        log_level="info",
    )


@serve_app.command("keys")
def serve_keys(
    count: int = typer.Option(1, help="How many keys to mint."),
    tier: str = typer.Option("default", help="Rate-limit tier for the key."),
) -> None:
    """Mint API keys for the gateway and print the hashed entries to store."""
    from dllm.serve.gateway.auth import mint_keys

    for entry in mint_keys(count, tier=tier):
        typer.echo(json.dumps(entry))


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #
@bench_app.command("run")
def bench_run(
    config: str = ConfigOpt,
    set_: list[str] = SetOpt,
    base_url: str = typer.Option(None, help="Endpoint to benchmark. Default: the gateway."),
    concurrency: str = typer.Option("1,2,4,8,16,32", help="Comma-separated concurrency sweep."),
    requests_per_level: int = typer.Option(48, help="Requests at each concurrency level."),
    out: str = typer.Option("artifacts/bench/latest.json", help="Where to write raw results."),
) -> None:
    """Measure throughput, TTFT and latency percentiles across a concurrency sweep."""
    from dllm.bench.loadgen import run_benchmark

    cfg = load_config(config, set_)
    levels = [int(x) for x in concurrency.split(",") if x.strip()]
    report = run_benchmark(
        cfg,
        base_url=base_url,
        levels=levels,
        requests_per_level=requests_per_level,
        out_path=ROOT / out,
    )
    typer.echo(json.dumps(report["summary"], indent=2))


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("interrupted", err=True)
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
