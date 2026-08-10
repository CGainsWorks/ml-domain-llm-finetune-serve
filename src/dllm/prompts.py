"""Prompt construction, shared by training, evaluation and serving.

This module exists so there is exactly one definition of what a prompt looks
like. If the training prompt and the eval prompt drift apart the fine-tune
appears to fail; if the eval prompt and the served prompt drift apart the
scorecard stops describing production. Every one of those three paths imports
from here.

Prompt formats are versioned. Changing ``SQL_PROMPT_VERSION`` invalidates
comparability with older scorecards, and the version is written into every
scorecard so a mismatch is visible rather than mysterious.
"""

from __future__ import annotations

from typing import Literal, TypedDict

SQL_PROMPT_VERSION = "sql_v1"
GUARD_PROMPT_VERSION = "guard_v1"


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


SQL_SYSTEM = (
    "You are a SQL generator. Given a database schema and a question, reply with "
    "a single valid SQLite SELECT query that answers it. Output only the query. "
    "No explanation, no markdown, no trailing semicolon."
)

SQL_USER_TEMPLATE = """Schema:
{context}

Question: {question}

SQL:"""

GUARD_SYSTEM = "You are a helpful assistant. Answer the question directly and concisely."


def sql_messages(question: str, context: str) -> list[Message]:
    """Chat messages for a text-to-SQL item."""
    return [
        {"role": "system", "content": SQL_SYSTEM},
        {
            "role": "user",
            "content": SQL_USER_TEMPLATE.format(context=context.strip(), question=question.strip()),
        },
    ]


def sql_target(answer: str) -> str:
    """The assistant turn a training example should teach."""
    return answer.strip().rstrip(";").strip()


def guard_messages(instruction: str, context: str = "") -> list[Message]:
    """Chat messages for the general-ability guard task."""
    content = (
        f"{instruction.strip()}\n\n{context.strip()}".strip() if context else instruction.strip()
    )
    return [
        {"role": "system", "content": GUARD_SYSTEM},
        {"role": "user", "content": content},
    ]


def render_fallback_chat(messages: list[Message]) -> str:
    """Plain-text rendering for tokenizers with no chat template.

    Only used as a fallback. Every model in ``configs/model`` ships a chat
    template; this keeps the harness from hard-crashing on one that does not.
    """
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    parts.append("<|assistant|>\n")
    return "\n".join(parts)
