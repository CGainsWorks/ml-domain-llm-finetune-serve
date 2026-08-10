"""Merge verification maths.

The first version of this check compared *decoded text* and rejected a
numerically exact merge, because greedy decoding turns a 1e-3 logit difference
at a near-tie into a completely different string. The replacement compares
logits under teacher forcing, which is why these tests exist: the check is only
worth having if it fires on a broken merge and stays quiet on a sound one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="verification operates on tensors")

from dllm.merge.merge_adapter import (  # noqa: E402
    MIN_MERGE_AGREEMENT,
    WARN_EXPORT_AGREEMENT,
    compare_logits,
)


def logits(rows: list[list[float]]) -> torch.Tensor:
    """One probe: shape (1, positions, vocab)."""
    return torch.tensor([rows], dtype=torch.float32)


class TestCompareLogits:
    def test_identical_tensors_agree_perfectly(self) -> None:
        a = logits([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        out = compare_logits([a], [a.clone()])
        assert out["argmax_agreement"] == 1.0
        assert out["max_abs_logit_diff"] == 0.0
        assert out["positions_compared"] == 2

    def test_tiny_perturbation_keeps_the_argmax(self) -> None:
        """The case the old text-comparison check got wrong."""
        a = logits([[1.0, 5.0, 3.0], [9.0, 1.0, 2.0]])
        b = a + 1e-3
        out = compare_logits([a], [b])
        assert out["argmax_agreement"] == 1.0
        assert out["max_abs_logit_diff"] == pytest.approx(1e-3, abs=1e-6)

    def test_flipped_argmax_is_detected(self) -> None:
        a = logits([[1.0, 5.0, 3.0]])
        b = logits([[9.0, 5.0, 3.0]])
        out = compare_logits([a], [b])
        assert out["argmax_agreement"] == 0.0
        assert out["max_abs_logit_diff"] == pytest.approx(8.0)

    def test_partial_agreement_is_a_fraction_of_positions(self) -> None:
        a = logits([[1.0, 5.0], [5.0, 1.0], [5.0, 1.0], [5.0, 1.0]])
        b = logits([[9.0, 5.0], [5.0, 1.0], [5.0, 1.0], [5.0, 1.0]])
        assert compare_logits([a], [b])["argmax_agreement"] == 0.75

    def test_aggregates_across_probes(self) -> None:
        a1 = logits([[1.0, 5.0], [5.0, 1.0]])
        a2 = logits([[1.0, 5.0], [5.0, 1.0]])
        out = compare_logits([a1, a2], [a1.clone(), a2.clone()])
        assert out["positions_compared"] == 4
        assert out["argmax_agreement"] == 1.0

    def test_mismatched_probe_counts_raise(self) -> None:
        a = logits([[1.0, 2.0]])
        with pytest.raises(ValueError):
            compare_logits([a, a], [a])

    def test_near_tie_does_not_hide_a_large_logit_shift(self) -> None:
        """Agreement alone is not enough; the magnitude is reported too."""
        a = logits([[1.0, 5.0]])
        b = logits([[1.0, 500.0]])
        out = compare_logits([a], [b])
        assert out["argmax_agreement"] == 1.0
        assert out["max_abs_logit_diff"] == pytest.approx(495.0)


class TestThresholds:
    def test_merge_gate_is_strict(self) -> None:
        """Same arithmetic, same precision: it should be essentially exact."""
        assert MIN_MERGE_AGREEMENT >= 0.999

    def test_export_warning_is_looser_than_the_merge_gate(self) -> None:
        """Reduced precision moving near-ties is expected, not a defect."""
        assert WARN_EXPORT_AGREEMENT < MIN_MERGE_AGREEMENT

    def test_bf16_cast_observed_in_practice_would_not_fail_the_merge_gate(self) -> None:
        """Regression guard for the real numbers from the first full merge.

        Measured on Qwen2.5-1.5B: the merge itself agreed 1.000000 in fp32,
        while casting the result to bf16 agreed 0.962. Gating the merge on the
        bf16 figure rejected a numerically exact merge.
        """
        measured_merge_fp32 = 1.0
        measured_export_bf16 = 0.962025
        assert measured_merge_fp32 >= MIN_MERGE_AGREEMENT
        assert measured_export_bf16 < MIN_MERGE_AGREEMENT, (
            "this is precisely why the export cast must not be gated on the merge threshold"
        )
        assert measured_export_bf16 >= WARN_EXPORT_AGREEMENT
