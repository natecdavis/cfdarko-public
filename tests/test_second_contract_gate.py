"""Tests for the second-contract label gate (defect fixes + replacement).

The gate is NOT enabled in production — the Stage-1 A/B failed its
pre-registered adoption bar (significant losses >= significant wins in every
arm). These tests pin the behaviour anyway so the harness stays trustworthy if
it is revisited, and so the legacy path can never silently drift.
"""

import numpy as np
import pandas as pd
import pytest

from config import TAG_MIN_APY_CAP_PCT, SECOND_CONTRACT_MAX_YEARS_AFTER_DRAFT
from src.scrapers.nfl_data import compute_second_contract_label


def _c(pfr_id, year_signed, draft_year, apy_cap_pct, years=3,
       draft_round=3, guaranteed=None, value=None):
    """One contract row. `guaranteed`/`value` default to a non-tag shape."""
    val = value if value is not None else apy_cap_pct * years * 100
    gtd = guaranteed if guaranteed is not None else val * 0.3
    return {"pfr_id": pfr_id, "year_signed": year_signed,
            "draft_year": draft_year, "apy_cap_pct": apy_cap_pct,
            "years": years, "draft_round": draft_round,
            "guaranteed": gtd, "value": val}


VET_MIN = pd.Series({y: 0.003 for y in range(2010, 2027)})


class TestLegacyBehaviour:
    def test_rookie_deal_excluded(self):
        df = pd.DataFrame([_c("A", 2018, 2018, 0.02)])
        out = compute_second_contract_label(df)
        assert out.loc["A", "label_apy_raw"] == 0.0

    def test_fifth_year_option_excluded(self):
        df = pd.DataFrame([_c("A", 2022, 2018, 0.05, years=1, draft_round=1)])
        out = compute_second_contract_label(df)
        assert out.loc["A", "label_apy_raw"] == 0.0

    def test_earliest_qualifying_wins(self):
        df = pd.DataFrame([_c("A", 2023, 2018, 0.09),
                           _c("A", 2021, 2018, 0.02)])
        out = compute_second_contract_label(df)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.02)

    def test_no_qualifying_contract_is_zero_not_missing(self):
        df = pd.DataFrame([_c("A", 2018, 2018, 0.02)])
        out = compute_second_contract_label(df)
        assert out.loc["A", "label_apy_raw"] == 0.0
        assert out.loc["A", "qualifies_role"] == 0.0

    def test_role_requires_positive_value(self):
        """Legacy semantics were `contract_label[pfr] > 0`; OTC does carry
        zero-APY rows, and those must not count as a second contract."""
        df = pd.DataFrame([_c("A", 2021, 2018, 0.0)])
        out = compute_second_contract_label(df)
        assert out.loc["A", "qualifies_role"] == 0.0

    def test_legacy_does_not_apply_any_gate(self):
        """A minimum-salary deal is a legacy second contract."""
        df = pd.DataFrame([_c("A", 2021, 2018, 0.001)])
        out = compute_second_contract_label(df)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.001)


class TestReplacementGate:
    def test_below_replacement_deal_is_rejected(self):
        df = pd.DataFrame([_c("A", 2021, 2018, 0.001)])
        out = compute_second_contract_label(df, replacement_gate=True,
                                             vet_min=VET_MIN, margin=1.25)
        assert out.loc["A", "label_apy_raw"] == 0.0

    def test_clearing_deal_is_kept_and_rebased(self):
        df = pd.DataFrame([_c("A", 2021, 2018, 0.05)])
        out = compute_second_contract_label(df, replacement_gate=True,
                                             vet_min=VET_MIN, margin=1.25)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.05)
        # 0.05 - (0.003 * 1.25)
        assert out.loc["A", "label_apy_over_repl"] == pytest.approx(0.05 - 0.00375)

    def test_over_replacement_is_floored_at_zero(self):
        df = pd.DataFrame([_c("A", 2021, 2018, 0.05)])
        out = compute_second_contract_label(df, replacement_gate=True,
                                             vet_min=VET_MIN, margin=1.25)
        assert out["label_apy_over_repl"].min() >= 0.0

    def test_earliest_CLEARING_deal_wins(self):
        """A futures deal should not pre-empt the real contract — this is the
        Adam Thielen case (0.003 futures vs 0.029 real)."""
        df = pd.DataFrame([_c("A", 2019, 2017, 0.003),
                           _c("A", 2021, 2017, 0.029)])
        out = compute_second_contract_label(df, replacement_gate=True,
                                             vet_min=VET_MIN, margin=1.25)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.029)

    def test_out_of_window_deal_rejected(self):
        far = 2018 + SECOND_CONTRACT_MAX_YEARS_AFTER_DRAFT + 1
        df = pd.DataFrame([_c("A", far, 2018, 0.09)])
        out = compute_second_contract_label(df, replacement_gate=True,
                                             vet_min=VET_MIN)
        assert out.loc["A", "label_apy_raw"] == 0.0

    def test_missing_draft_year_survives_the_window(self):
        df = pd.DataFrame([_c("A", 2021, np.nan, 0.05)])
        out = compute_second_contract_label(df, replacement_gate=True,
                                             vet_min=VET_MIN)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.05)


class TestTagHandling:
    """A franchise tag is evidence of quality, so it is DE-PREFERRED in
    selection but never allowed to delete the player."""

    def _tag(self, pfr_id, year, draft_year, apy=0.15):
        return _c(pfr_id, year, draft_year, apy, years=1,
                  value=100.0, guaranteed=99.0)

    def test_tag_loses_to_a_real_deal(self):
        """The Dak Prescott case: tag at 0.158 then extension at 0.219."""
        df = pd.DataFrame([self._tag("A", 2020, 2016, 0.158),
                           _c("A", 2021, 2016, 0.219)])
        out = compute_second_contract_label(df, defect_fixes=True)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.219)

    def test_tag_is_kept_when_it_is_the_only_deal(self):
        """Regression: zeroing a tag-only player labels a franchised — i.e.
        clearly valuable — player as a washout. The first 3-arm A/B lost on
        exactly this."""
        df = pd.DataFrame([self._tag("A", 2020, 2016, 0.158)])
        out = compute_second_contract_label(df, defect_fixes=True)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.158)
        assert out.loc["A", "qualifies_role"] == 1.0

    def test_low_value_one_year_deal_is_not_a_tag(self):
        """The tag heuristic requires a high cap share; a fully-guaranteed
        minimum deal must not trip it."""
        low = TAG_MIN_APY_CAP_PCT / 10
        df = pd.DataFrame([_c("A", 2020, 2016, low, years=1,
                              value=1.0, guaranteed=1.0),
                           _c("A", 2022, 2016, 0.08)])
        out = compute_second_contract_label(df, defect_fixes=True)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(low)

    def test_partially_guaranteed_one_year_deal_is_not_a_tag(self):
        df = pd.DataFrame([_c("A", 2020, 2016, 0.15, years=1,
                              value=100.0, guaranteed=10.0),
                           _c("A", 2022, 2016, 0.20)])
        out = compute_second_contract_label(df, defect_fixes=True)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.15)


class TestFlagSeparation:
    """defect_fixes and replacement_gate must be independently controllable —
    bundling them made the first A/B uninterpretable."""

    def test_defect_fixes_alone_does_not_rebase(self):
        df = pd.DataFrame([_c("A", 2021, 2018, 0.05)])
        out = compute_second_contract_label(df, defect_fixes=True,
                                             replacement_gate=False)
        assert out.loc["A", "label_apy_over_repl"] == pytest.approx(0.05)

    def test_defect_fixes_alone_keeps_minimum_deals(self):
        df = pd.DataFrame([_c("A", 2021, 2018, 0.001)])
        out = compute_second_contract_label(df, defect_fixes=True,
                                             replacement_gate=False)
        assert out.loc["A", "label_apy_raw"] == pytest.approx(0.001)

    def test_replacement_gate_defaults_to_enabling_defect_fixes(self):
        far = 2018 + SECOND_CONTRACT_MAX_YEARS_AFTER_DRAFT + 1
        df = pd.DataFrame([_c("A", far, 2018, 0.09)])
        out = compute_second_contract_label(df, replacement_gate=True,
                                             vet_min=VET_MIN)
        assert out.loc["A", "label_apy_raw"] == 0.0


class TestFrameContract:
    def test_returns_expected_columns(self):
        df = pd.DataFrame([_c("A", 2021, 2018, 0.05)])
        out = compute_second_contract_label(df)
        for col in ["label_apy_raw", "label_apy_over_repl", "qualifies_role",
                    "max_apy_in_window", "year_signed"]:
            assert col in out.columns

    def test_every_input_player_gets_a_row(self):
        df = pd.DataFrame([_c("A", 2018, 2018, 0.02), _c("B", 2021, 2018, 0.05)])
        out = compute_second_contract_label(df)
        assert set(out.index) == {"A", "B"}

    def test_max_in_window_is_gate_independent(self):
        """The diagnostic must describe the player, not the arm, so it stays
        comparable across A/B arms."""
        df = pd.DataFrame([_c("A", 2021, 2018, 0.001), _c("A", 2022, 2018, 0.09)])
        legacy = compute_second_contract_label(df)
        gated = compute_second_contract_label(df, replacement_gate=True,
                                               vet_min=VET_MIN)
        assert legacy.loc["A", "max_apy_in_window"] == \
               gated.loc["A", "max_apy_in_window"] == pytest.approx(0.09)

    def test_empty_contracts_returns_zeros_not_crash(self):
        df = pd.DataFrame(columns=["pfr_id", "year_signed", "draft_year",
                                   "apy_cap_pct", "years", "draft_round",
                                   "guaranteed", "value"])
        out = compute_second_contract_label(df)
        assert len(out) == 0
