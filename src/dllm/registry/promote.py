"""Register a model version in MLflow and promote it only if the gate passes.

The order matters. The version is registered *first*, with its scorecard and
gate report attached, and only then does the alias move. That means a blocked
candidate still leaves a permanent, inspectable record of what it scored and
which check it failed -- which is exactly what you want three weeks later when
someone asks why v4 was never deployed.

``deployment_target`` is the only alias the deploy workflow reads. Nothing
except a passing gate ever moves it.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from dllm.config import PipelineConfig
from dllm.eval.gate import decide
from dllm.eval.scorecard import Scorecard, render_comparison
from dllm.paths import ROOT

log = logging.getLogger(__name__)

DEPLOY_ALIAS = "deployment_target"


def promote(
    cfg: PipelineConfig,
    *,
    candidate_path: str,
    baseline_path: str | None = None,
    model_dir: str | None = None,
    name: str = "domain-sql",
    alias: str = "Staging",
) -> dict[str, Any]:
    """Evaluate the gate, register the version, and move aliases if it passed."""
    import mlflow
    from mlflow.tracking import MlflowClient

    candidate = Scorecard.load(candidate_path)
    base_path = baseline_path or cfg.eval.baseline_scorecard
    baseline = Scorecard.load(base_path) if Path(base_path).is_file() else None

    decision = decide(candidate, cfg.eval.thresholds, baseline)

    mlflow.set_tracking_uri(cfg.train.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.train.mlflow_experiment)
    client = MlflowClient()

    report_dir = ROOT / "artifacts" / "gate"
    report_dir.mkdir(parents=True, exist_ok=True)
    gate_report = report_dir / "gate_report.md"
    body = decision.render()
    if baseline is not None:
        body += "\n" + render_comparison(baseline, candidate)
    gate_report.write_text(body, encoding="utf-8")
    (report_dir / "gate_decision.json").write_text(
        json.dumps(decision.as_dict(), indent=2) + "\n", encoding="utf-8"
    )

    with mlflow.start_run(run_name=f"promote-{candidate.label}") as run:
        mlflow.log_metrics(
            {k: v for k, v in candidate.metrics.items() if isinstance(v, int | float)}
        )
        mlflow.log_param("gate_passed", decision.promote)
        mlflow.log_param("candidate_label", candidate.label)
        mlflow.log_param("baseline_label", baseline.label if baseline else "none")
        mlflow.log_artifact(str(gate_report))
        mlflow.log_artifact(str(Path(candidate_path)))
        if baseline is not None and Path(base_path).is_file():
            mlflow.log_artifact(str(Path(base_path)))

        version = None
        if model_dir:
            source = Path(model_dir)
            if not source.is_dir():
                raise FileNotFoundError(f"model directory not found: {source}")
            # Log the merged checkpoint as an artifact of this run, then
            # register that artifact path as a model version.
            mlflow.log_artifacts(str(source), artifact_path="model")
            model_uri = f"runs:/{run.info.run_id}/model"
            # MLflow has no idempotent create; the second call raises
            # RESOURCE_ALREADY_EXISTS, which is the expected path here.
            with contextlib.suppress(Exception):
                client.create_registered_model(name)
            mv = client.create_model_version(name=name, source=model_uri, run_id=run.info.run_id)
            version = mv.version
            client.set_model_version_tag(name, version, "gate_passed", str(decision.promote))
            client.set_model_version_tag(name, version, "candidate_label", candidate.label)
            for metric, value in candidate.metrics.items():
                client.set_model_version_tag(name, version, f"metric.{metric}", f"{value}")

            if decision.promote:
                client.set_registered_model_alias(name, alias, version)
                client.set_registered_model_alias(name, DEPLOY_ALIAS, version)
                log.info(
                    "version %s promoted: aliases %s and %s now point at it",
                    version,
                    alias,
                    DEPLOY_ALIAS,
                )
            else:
                client.set_registered_model_alias(name, "blocked", version)
                log.error(
                    "version %s BLOCKED by the eval gate; deployment alias unchanged", version
                )

        result = {
            "promoted": decision.promote,
            "registered_model": name if model_dir else None,
            "version": version,
            "alias": alias if decision.promote else "blocked",
            "mlflow_run_id": run.info.run_id,
            "gate_report": str(gate_report),
            "failures": [c.message for c in decision.failures],
            "blocking_errors": decision.blocking_errors,
        }

    if not decision.promote:
        log.error(
            "promotion blocked:\n  - %s",
            "\n  - ".join(result["failures"] + result["blocking_errors"]),
        )
    return result
