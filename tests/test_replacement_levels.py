"""Tests for replacement-value derivation and the step-10 output columns."""

import numpy as np
import pandas as pd
import pytest

from config import (
    REPLACEMENT_MARGIN,
    REPLACEMENT_POS_MIN_N,
    VET_MIN_FALLBACK_PCT,
    OTC_POSITION_TO_GROUP,
)
from src.scrapers.nfl_data import (
    derive_vet_min_apy_pct,
    vet_min_for_year,
    compute_replacement_levels,
)


def _synth_contracts(year=2025, cap=280.0, min_apy=0.840, n_min=60,
                     n_other=200, seed=0) -> pd.DataFrame:
    """Build a contract frame with a sharp minimum-salary mode plus a spread of
    negotiated deals, mimicking the OTC structure."""
    rng = np.random.default_rng(seed)
    rows = []
    # Sharp mode at the minimum: one-year, non-rookie.
    for i in range(n_min):
        rows.append({"year_signed": year, "years": 1, "apy": min_apy,
                     "draft_year": year - 4, "position": "WR"})
    # Prorated deals well below the minimum — must be excluded.
    for i in range(40):
        rows.append({"year_signed": year, "years": 1, "apy": min_apy * 0.25,
                     "draft_year": year - 4, "position": "WR"})
    # Negotiated multi-year deals across positions.
    for i in range(n_other):
        pos = ["WR", "QB", "IDL", "ED", "LT", "CB"][i % 6]
        rows.append({"year_signed": year, "years": int(rng.integers(2, 6)),
                     "apy": float(rng.uniform(1.0, 40.0)),
                     "draft_year": year - 4, "position": pos})
    df = pd.DataFrame(rows)
    df["apy_cap_pct"] = (df["apy"] / cap).round(3)
    df["implied_cap"] = np.where(df["apy_cap_pct"] > 0,
                                 df["apy"] / df["apy_cap_pct"], np.nan)
    df["position_group"] = df["position"].map(OTC_POSITION_TO_GROUP)
    df["guaranteed"] = 0.0
    df["value"] = df["apy"] * df["years"]
    df["gsis_id"] = [f"P{i:05d}" for i in range(len(df))]
    return df


class TestDeriveVetMin:
    def test_extracts_the_sharp_minimum_mode(self):
        df = _synth_contracts(year=2025, cap=280.0, min_apy=0.840)
        s = derive_vet_min_apy_pct(df)
        assert 2025 in s.index
        # Should recover 0.840/280 ≈ 0.003, not the prorated 0.21/280.
        assert s.loc[2025] == pytest.approx(0.840 / 280.0, rel=0.05)

    def test_prorated_deals_are_excluded(self):
        """The prorated cluster is more numerous than nothing but sits below
        the proration floor; picking it would roughly quarter the line."""
        df = _synth_contracts(min_apy=0.840, cap=280.0)
        s = derive_vet_min_apy_pct(df)
        assert s.loc[2025] > 0.002

    def test_sparse_years_are_backfilled(self):
        a = _synth_contracts(year=2025, cap=280.0, min_apy=0.840)
        b = _synth_contracts(year=2024, cap=255.0, min_apy=0.795, n_min=2,
                             n_other=3, seed=1)
        s = derive_vet_min_apy_pct(pd.concat([a, b], ignore_index=True))
        # 2024 has too few rows for a sharp mode → backfilled, not missing.
        assert 2024 in s.index
        assert np.isfinite(s.loc[2024])

    def test_empty_input_returns_empty(self):
        s = derive_vet_min_apy_pct(pd.DataFrame(
            columns=["year_signed", "apy", "apy_cap_pct", "implied_cap",
                     "years", "draft_year"]))
        assert len(s) == 0


class TestVetMinLookup:
    def test_exact_year(self):
        s = pd.Series({2020: 0.003, 2021: 0.004})
        assert vet_min_for_year(s, 2021) == 0.004

    def test_out_of_range_uses_nearest(self):
        """Bridge classes reach back to 2004 but OTC starts ~2007, so
        out-of-range lookups are the normal case, not an error."""
        s = pd.Series({2010: 0.003, 2020: 0.005})
        assert vet_min_for_year(s, 2004) == 0.003
        assert vet_min_for_year(s, 2030) == 0.005

    def test_empty_series_falls_back_to_constant(self):
        assert vet_min_for_year(pd.Series(dtype=float), 2020) == VET_MIN_FALLBACK_PCT

    def test_nan_year_falls_back(self):
        s = pd.Series({2020: 0.003})
        assert vet_min_for_year(s, float("nan")) == VET_MIN_FALLBACK_PCT


class TestComputeReplacementLevels:
    def test_emits_both_scopes(self):
        lv = compute_replacement_levels(_synth_contracts())
        assert set(lv["scope"]) == {"global_year", "position"}
        assert {"scope", "key", "replacement_apy_pct", "base_apy_pct",
                "n_contracts", "source"} <= set(lv.columns)

    def test_global_line_applies_the_margin(self):
        lv = compute_replacement_levels(_synth_contracts())
        g = lv[lv["scope"] == "global_year"].iloc[0]
        assert g["replacement_apy_pct"] == pytest.approx(
            g["base_apy_pct"] * REPLACEMENT_MARGIN)

    def test_thin_position_cohorts_fall_back(self):
        """A position with fewer than REPLACEMENT_POS_MIN_N multi-year deals
        must not get its own noisy quantile."""
        lv = compute_replacement_levels(_synth_contracts(n_other=12))
        pos = lv[lv["scope"] == "position"]
        thin = pos[pos["n_contracts"] < REPLACEMENT_POS_MIN_N]
        assert (thin["source"] == "global_fallback").all()

    def test_levels_are_positive_and_finite(self):
        lv = compute_replacement_levels(_synth_contracts())
        assert (lv["replacement_apy_pct"] > 0).all()
        assert np.isfinite(lv["replacement_apy_pct"]).all()

    def test_position_line_is_not_quantization_collapsed(self):
        """apy_cap_pct is quantized to 3 decimals in the OTC feed; computing the
        per-position quantile off it collapses every group onto one value. The
        implementation must use the continuous APY share instead."""
        lv = compute_replacement_levels(_synth_contracts(n_other=600, seed=7))
        pos = lv[(lv["scope"] == "position")
                 & (lv["source"] == "position_quantile")]
        assert pos["replacement_apy_pct"].nunique() > 1


class TestOtcPositionMap:
    def test_all_targets_are_real_position_groups(self):
        from config import POSITION_GROUPS
        assert set(OTC_POSITION_TO_GROUP.values()) <= set(POSITION_GROUPS)

    def test_long_snapper_is_not_mapped(self):
        assert "LS" not in OTC_POSITION_TO_GROUP

    def test_edge_is_distinct_from_dl(self):
        assert OTC_POSITION_TO_GROUP["ED"] == "EDGE"
        assert OTC_POSITION_TO_GROUP["IDL"] == "DL"


class TestOutputColumns:
    """_add_replacement_columns behaviour, with the on-disk level artifact
    stubbed out so the test does not depend on pipeline state."""

    LEVELS = {"WR": 0.0037, "QB": 0.0069}

    def _mod(self, monkeypatch):
        import sys, types, importlib.util
        from pathlib import Path
        mod_path = Path(__file__).parent.parent / "pipeline" / "10_output.py"
        spec = importlib.util.spec_from_file_location("output_10_repl", mod_path)
        mod = types.ModuleType("output_10_repl")
        mod.__file__ = str(mod_path)
        pff_stub = types.ModuleType("src.scrapers.pff")
        pff_stub.load_pff_career_features = lambda *a, **kw: pd.DataFrame()
        sys.modules.setdefault("src.scrapers.pff", pff_stub)
        spec.loader.exec_module(mod)
        monkeypatch.setattr(mod, "_load_replacement_levels",
                            lambda: dict(self.LEVELS))
        return mod

    def _snap(self):
        return pd.DataFrame({
            "position_group": ["WR", "WR", "QB", "QB"],
            "impact_apy_score": [0.0010, 0.0500, 0.0020, 0.0600],
            "raw_impact_score": [0.0011, 0.0501, 0.0021, 0.0601],
        })

    def test_below_replacement_is_negative_not_clipped(self, monkeypatch):
        """Flooring at zero was measured to collapse ~95% of predictions into a
        single tie, destroying ordering — the column must stay signed."""
        mod = self._mod(monkeypatch)
        out = mod._add_replacement_columns(self._snap())
        ior = out["impact_over_replacement"]
        assert ior.iloc[0] < 0          # WR at 0.001 is below the 0.0037 line
        assert ior.iloc[1] > 0          # WR at 0.050 is well above
        assert ior.iloc[2] < 0          # QB at 0.002 is below the 0.0069 line

    def test_emits_expected_columns(self, monkeypatch):
        mod = self._mod(monkeypatch)
        out = mod._add_replacement_columns(self._snap())
        for col in ["replacement_apy_pct", "impact_over_replacement",
                    "impact_over_replacement_dollars", "ior_rank"]:
            assert col in out.columns

    def test_impact_score_is_left_untouched(self, monkeypatch):
        """The adjustment is additive: impact_score stays the raw APY%cap
        projection so the impact_apy_dollars identity holds."""
        mod = self._mod(monkeypatch)
        snap = self._snap()
        snap["impact_score"] = snap["impact_apy_score"]
        before = snap["impact_score"].copy()
        out = mod._add_replacement_columns(snap)
        pd.testing.assert_series_equal(out["impact_score"], before)

    def test_subtraction_preserves_within_position_order(self, monkeypatch):
        """A per-position constant shift must not reorder within a position."""
        mod = self._mod(monkeypatch)
        out = mod._add_replacement_columns(self._snap())
        for pos, grp in out.groupby("position_group"):
            assert list(grp["impact_apy_score"].rank()) == \
                   list(grp["impact_over_replacement"].rank())

    def test_qb_is_penalised_more_than_wr(self, monkeypatch):
        """The whole cross-position point: identical raw scores should not
        yield identical above-replacement values when replacement differs."""
        mod = self._mod(monkeypatch)
        snap = pd.DataFrame({
            "position_group": ["WR", "QB"],
            "impact_apy_score": [0.05, 0.05],
            "raw_impact_score": [0.05, 0.05],
        })
        out = mod._add_replacement_columns(snap)
        assert out["impact_over_replacement"].iloc[0] > \
               out["impact_over_replacement"].iloc[1]

    def test_missing_artifact_degrades_gracefully(self, monkeypatch):
        mod = self._mod(monkeypatch)
        monkeypatch.setattr(mod, "_load_replacement_levels", dict)
        snap = self._snap()
        out = mod._add_replacement_columns(snap)
        assert "impact_over_replacement" not in out.columns
