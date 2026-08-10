"""Evaluation harness: tasks, backends, scoring, scorecards and the gate."""

from dllm.eval.gate import GateDecision, decide
from dllm.eval.scorecard import Scorecard, render_comparison, render_single

__all__ = ["GateDecision", "Scorecard", "decide", "render_comparison", "render_single"]
