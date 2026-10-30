"""Adapter release packaging: what goes in, what stays out, and the hashes."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from dllm.config import load_config
from dllm.merge.package import package_adapter, sha256_file
from dllm.paths import CONFIGS

CFG = load_config(CONFIGS / "base.yaml")


@pytest.fixture
def adapter(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    d = run / "adapter"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"\x00weights\x01" * 100)
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": CFG.model.id, "revision": None, "r": 16})
    )
    (d / "tokenizer.json").write_text("{}")
    (d / "tokenizer_config.json").write_text("{}")
    (d / "chat_template.jinja").write_text("{{ messages }}")
    # Things that must NOT be shipped.
    (d / "training_args.bin").write_bytes(b"pickle")
    (d / "README.md").write_text("PEFT template")
    (run / "train_summary.json").write_text(
        json.dumps(
            {
                "run_id": "abc",
                "output_dir": "C:\\Users\\someone\\run",
                "adapter_dir": "/home/someone/run/adapter",
                "gpu_hours": 1.1487,
                "global_step": 2500,
            }
        )
    )
    return d


def _package(adapter: Path, out: Path) -> dict:
    return package_adapter(
        CFG,
        adapter_dir=adapter,
        out_dir=out,
        name="pkg",
        version="v0",
        train_summary=adapter.parent / "train_summary.json",
    )


def test_ships_weights_licence_and_notice_but_not_the_pickle(adapter: Path, tmp_path: Path) -> None:
    m = _package(adapter, tmp_path / "out")
    names = {f["name"] for f in m["files"]}
    assert {"adapter_model.safetensors", "adapter_config.json", "LICENSE", "NOTICE.md"} <= names
    assert "training_args.bin" not in names
    assert "README.md" not in names


def test_tokenizer_comes_from_the_pinned_base_not_the_package(
    adapter: Path, tmp_path: Path
) -> None:
    m = _package(adapter, tmp_path / "out")
    names = {f["name"] for f in m["files"]}
    assert "tokenizer.json" not in names
    assert "tokenizer_config.json" not in names
    assert "chat_template.jinja" in names
    assert m["tokenizer"]["id"] == CFG.model.id
    assert m["tokenizer"]["revision"] == CFG.model.revision


def test_hashes_match_the_files(adapter: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    m = _package(adapter, out)
    for f in m["files"]:
        p = out / "pkg-v0" / f["name"]
        assert f["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        assert f["size_bytes"] == p.stat().st_size
    archive = out / m["archive"]["name"]
    assert m["archive"]["sha256"] == sha256_file(archive)
    assert (out / "SHA256SUMS").read_text() == f"{m['archive']['sha256']}  pkg-v0.tar.gz\n"


def test_archive_is_deterministic(adapter: Path, tmp_path: Path) -> None:
    a = _package(adapter, tmp_path / "a")["archive"]["sha256"]
    b = _package(adapter, tmp_path / "b")["archive"]["sha256"]
    assert a == b


def test_archive_holds_exactly_the_listed_files(adapter: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    m = _package(adapter, out)
    with tarfile.open(out / m["archive"]["name"]) as tar:
        members = sorted(t.name for t in tar.getmembers())
    assert members == sorted(f"pkg-v0/{f['name']}" for f in m["files"])


def test_base_revision_is_pinned_in_the_shipped_config(adapter: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _package(adapter, out)
    shipped = json.loads((out / "pkg-v0" / "adapter_config.json").read_text())
    assert shipped["revision"] == CFG.model.revision


def test_manifest_records_licences_and_no_paths(adapter: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    m = _package(adapter, out)
    assert m["licence"] == "Apache-2.0"
    assert m["base_model"]["licence"] == CFG.model.licence
    assert {d["licence"] for d in m["training_data"]} >= {"CC-BY-4.0", "CC-BY-SA-4.0"}
    assert m["training"] == {"run_id": "abc", "gpu_hours": 1.1487, "global_step": 2500}
    text = (out / "pkg-v0.manifest.json").read_text()
    assert "someone" not in text and "Users" not in text


def test_wrong_base_model_is_refused(adapter: Path, tmp_path: Path) -> None:
    cfg_path = adapter / "adapter_config.json"
    cfg_path.write_text(json.dumps({"base_model_name_or_path": "some/other-model"}))
    with pytest.raises(ValueError, match="some/other-model"):
        _package(adapter, tmp_path / "out")
