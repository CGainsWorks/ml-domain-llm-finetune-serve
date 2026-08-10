"""Registry of every dataset this repo is allowed to touch, with its licence.

Nothing downloads a dataset that is not declared here. The licence fields are
not decoration: ``dllm.data.licence_audit`` reads them, and CI fails the build
if a config selects a source whose terms do not permit the declared use.

Provenance matters as much as the top-level licence. ``sql-create-context`` is
published under CC-BY-4.0, but it is *derived* from WikiSQL (BSD-3-Clause) and
Spider (CC-BY-SA-4.0). The share-alike term on Spider propagates, so the
derived-data obligations are recorded per upstream, not just per dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Use = Literal["research_only", "commercial_ok", "commercial_with_conditions"]


@dataclass(frozen=True)
class Upstream:
    """A dataset that a source is derived from."""

    name: str
    licence: str
    url: str
    note: str = ""


@dataclass(frozen=True)
class DatasetSource:
    """A declared, licence-audited dataset."""

    key: str
    hf_id: str
    revision: str
    licence: str
    licence_url: str
    use: Use
    homepage: str
    citation: str
    fields: dict[str, str]
    upstreams: list[Upstream] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)
    split: str = "train"
    notes: str = ""

    @property
    def commercial_ok(self) -> bool:
        return self.use in {"commercial_ok", "commercial_with_conditions"}


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
SOURCES: dict[str, DatasetSource] = {
    # ---- Primary training source -------------------------------------------
    "sql_create_context": DatasetSource(
        key="sql_create_context",
        hf_id="b-mc2/sql-create-context",
        revision="9d80a6a118b838d9defc3798d659a54a2ac2ff37",
        licence="CC-BY-4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        use="commercial_with_conditions",
        homepage="https://huggingface.co/datasets/b-mc2/sql-create-context",
        citation=(
            "b-mc2 (2023). sql-create-context. Hugging Face Datasets. "
            "Built from WikiSQL and Spider."
        ),
        fields={
            "question": "natural language question",
            "context": "CREATE TABLE statement(s) defining the schema",
            "answer": "gold SQL query",
        },
        upstreams=[
            Upstream(
                name="WikiSQL",
                licence="BSD-3-Clause",
                url="https://github.com/salesforce/WikiSQL",
                note="Permissive. Attribution + licence text retention.",
            ),
            Upstream(
                name="Spider",
                licence="CC-BY-SA-4.0",
                url="https://yale-lily.github.io/spider",
                note=(
                    "Share-alike. Derived *datasets* must be redistributed under "
                    "CC-BY-SA-4.0. Model weights trained on it are not generally "
                    "treated as an adapted work, but this repo does not "
                    "redistribute the data, only the preparation script."
                ),
            ),
        ],
        obligations=[
            "Attribute b-mc2, WikiSQL (Salesforce) and Spider (Yale LILY).",
            "Do not redistribute prepared splits without CC-BY-SA-4.0 terms; "
            "this repo ships the preparation script instead of the data.",
            "State in the model card that training data derives from Spider.",
        ],
        notes=(
            "Chosen as the primary domain because correctness is machine-checkable: "
            "a predicted query can be executed against a materialised schema and "
            "compared to the gold result set. No LLM judge in the loop."
        ),
    ),
    # ---- Held-out / contrast eval source -----------------------------------
    "spider_dev": DatasetSource(
        key="spider_dev",
        hf_id="xlangai/spider",
        revision="main",
        licence="CC-BY-SA-4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
        use="commercial_with_conditions",
        homepage="https://yale-lily.github.io/spider",
        citation="Yu et al. (2018). Spider: A Large-Scale Human-Labeled Dataset for "
        "Complex and Cross-Domain Semantic Parsing and Text-to-SQL. EMNLP.",
        fields={"question": "question", "query": "gold SQL", "db_id": "database id"},
        obligations=[
            "Share-alike on any redistributed derivative dataset.",
            "Attribute Yale LILY.",
        ],
        split="validation",
        notes=(
            "Eval only, never trained on. Overlaps the upstream of the training "
            "set, so treat it as in-distribution rather than a clean holdout; "
            "the authoritative holdout is our own test split."
        ),
    ),
    # ---- General-ability guard ---------------------------------------------
    "dolly_guard": DatasetSource(
        key="dolly_guard",
        hf_id="databricks/databricks-dolly-15k",
        revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        licence="CC-BY-SA-3.0",
        licence_url="https://creativecommons.org/licenses/by-sa/3.0/",
        use="commercial_with_conditions",
        homepage="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        citation="Conover et al. (2023). Free Dolly. Databricks.",
        fields={
            "instruction": "instruction",
            "context": "optional context",
            "response": "reference answer",
        },
        obligations=["Attribute Databricks.", "Share-alike on redistributed derivatives."],
        notes=(
            "Used ONLY as a catastrophic-forgetting guard: a fine-tune that gets "
            "better at SQL but starts answering every general question with SQL "
            "must fail the gate. Not part of the training mix."
        ),
    ),
    # ---- Explicitly rejected ------------------------------------------------
    "alpaca_REJECTED": DatasetSource(
        key="alpaca_REJECTED",
        hf_id="tatsu-lab/alpaca",
        revision="main",
        licence="CC-BY-NC-4.0",
        licence_url="https://creativecommons.org/licenses/by-nc/4.0/",
        use="research_only",
        homepage="https://huggingface.co/datasets/tatsu-lab/alpaca",
        citation="Taori et al. (2023). Stanford Alpaca.",
        fields={"instruction": "instruction", "input": "input", "output": "output"},
        obligations=[
            "Non-commercial only.",
            "Generated with OpenAI models; subject to "
            "the OpenAI terms in force at generation time.",
        ],
        notes=(
            "Kept in the registry deliberately, as a REJECTED entry. It is the "
            "obvious default instruction set and it is non-commercial. Selecting "
            "it while data.require_commercial_licence is true fails the audit, "
            "which is the behaviour the audit exists to demonstrate."
        ),
    ),
}


def get_source(key: str) -> DatasetSource:
    try:
        return SOURCES[key]
    except KeyError:
        known = ", ".join(sorted(SOURCES))
        raise KeyError(f"unknown dataset source {key!r}. Declared sources: {known}") from None
