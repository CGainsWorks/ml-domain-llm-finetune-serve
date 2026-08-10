"""Turn prepared JSONL into the prompt/completion format TRL trains on.

The prompt is built by ``dllm.prompts.sql_messages`` -- the same function the
eval harness and the served gateway call. That is deliberate. A fine-tune
trained on one prompt and evaluated on another looks like a failed fine-tune,
and the bug is invisible because both halves work in isolation.

Emitting ``prompt``/``completion`` columns (rather than one pre-rendered text
field) lets TRL mask the prompt tokens out of the loss. For text-to-SQL the
answer is a small fraction of the sequence, so training on the full sequence
spends most of the gradient teaching the model to reproduce schemas it was
given anyway.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dllm.data.prepare import read_jsonl
from dllm.prompts import sql_messages, sql_target

log = logging.getLogger(__name__)


def to_prompt_completion(item: dict) -> dict[str, Any]:
    """One prepared row -> a conversational prompt/completion pair."""
    return {
        "prompt": sql_messages(item["question"], item["context"]),
        "completion": [{"role": "assistant", "content": sql_target(item["answer"])}],
    }


def build_datasets(
    processed_dir: Path,
    *,
    max_train: int | None = None,
    max_eval: int | None = None,
) -> tuple[Any, Any]:
    """Load the train and validation splits as Hugging Face datasets."""
    from datasets import Dataset

    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "validation.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"{train_path} missing. Run `dllm data prepare` first.")

    train_rows = [to_prompt_completion(r) for r in read_jsonl(train_path, max_train)]
    # The in-loop eval exists to catch divergence, not to produce the headline
    # number, so it is capped hard. A 500-item eval every 100 steps would cost
    # more GPU time than the training it is monitoring.
    eval_rows = (
        [to_prompt_completion(r) for r in read_jsonl(val_path, max_eval or 200)]
        if val_path.is_file()
        else []
    )

    log.info("train examples: %d, in-loop eval examples: %d", len(train_rows), len(eval_rows))
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows) if eval_rows else None
    return train_ds, eval_ds
