"""Trust gate for the PFR combine supplement.

Parses the cached PFR 2025 combine HTML the same way 00d0 parses 2026, then
diffs against the current nflverse-sourced 2025 rows in combine_metrics.parquet.
PASS criteria:
  - >= 95% of nflverse 2025 names match a PFR row (by normalized name)
  - >= 99% of matched-row numeric checks pass within tolerance

Run BEFORE re-running pipeline/00d_combine_ingestion.py. If this fails, do not
trust the PFR scrape for 2026 either.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from config import RAW_DIR, PROCESSED_DIR

# Import 00d0's parser via importlib (filename starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "_supp", REPO / "pipeline" / "00d0_pfr_combine_supplement.py"
)
_supp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_supp)

# Import 00d's name normalizer + ht parser the same way
_spec2 = importlib.util.spec_from_file_location(
    "_d", REPO / "pipeline" / "00d_combine_ingestion.py"
)
_d = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_d)

YEAR = 2025

# Per-field tolerances (PFR sometimes corrects bad numbers; small drift OK)
TOLERANCES = {
    "wt":         2.0,
    "forty":      0.02,
    "vertical":   0.5,
    "broad_jump": 1.0,
    "bench":      0.0,    # exact (integer reps)
    "ht":         0.0,    # exact (inches, after conversion)
}


def main() -> None:
    # ── Load PFR 2025 from cached HTML, run through the same parser as 00d0 ──
    html_path = RAW_DIR / "pfr_combine" / f"{YEAR}.html"
    if not html_path.exists():
        print(f"ERROR: {html_path} not found. Save PFR's {YEAR} combine page there first.")
        sys.exit(2)

    html = html_path.read_text(encoding="utf-8")
    df_raw = _supp.parse_pfr_combine_table(html)
    pfr_ids = _supp.extract_pfr_ids(html)
    pfr_df = _supp.normalize_to_nflverse_schema(df_raw, YEAR, pfr_ids)
    pfr_df["ht_in"] = pfr_df["ht"].apply(_d._ht_to_inches)
    pfr_df["_norm_name"] = pfr_df["player_name"].apply(_d._normalize_name)
    print(f"PFR {YEAR}: {len(pfr_df)} rows")

    # ── Load current nflverse 2025 from combine_metrics.parquet ─────────────────
    cm = pd.read_parquet(PROCESSED_DIR / "combine_metrics.parquet")
    cm_y = cm[cm["combine_draft_year"] == YEAR].copy()
    print(f"combine_metrics.parquet {YEAR}: {len(cm_y)} rows")

    # nflverse rows in combine_metrics use combine_<metric> column names.
    # Need to back-resolve player_name and a normalized key. They aren't kept in
    # combine_metrics.parquet (only player_id is) — re-derive by joining to bridge_players.
    bridge = pd.read_parquet(PROCESSED_DIR / "bridge_players.parquet",
                             columns=["player_id", "player_name"])
    bridge = bridge[bridge["player_id"].notna()].drop_duplicates("player_id")
    cm_y = cm_y.merge(bridge, on="player_id", how="left")
    cm_y["_norm_name"] = cm_y["player_name"].apply(_d._normalize_name)
    print(f"  with player_name resolved via bridge: {cm_y['player_name'].notna().sum()}/{len(cm_y)}")

    # ── Match by normalized name ────────────────────────────────────────────────
    merged = cm_y.merge(pfr_df, on="_norm_name", how="left", suffixes=("_nfl", "_pfr"))
    n_matched = merged["player_name_pfr"].notna().sum()
    match_rate = n_matched / len(cm_y) if len(cm_y) else 0.0
    print(f"\nName-match rate: {n_matched}/{len(cm_y)} = {match_rate:.1%}  (threshold ≥ 95%)")

    # ── Per-field numeric diff on matched rows ──────────────────────────────────
    matched = merged[merged["player_name_pfr"].notna()].copy()
    print(f"\nField-level diffs on {len(matched)} matched rows:")
    print(f"{'field':12s} {'tol':>8s} {'nfl_n':>6s} {'pfr_n':>6s} {'both':>6s} {'pass':>6s} {'fail':>6s} {'pass%':>7s}")

    total_checks = 0
    total_pass = 0
    fail_examples: list[str] = []

    for nfl_field, pfr_field in [
        ("combine_wt", "wt"), ("combine_forty", "forty"),
        ("combine_vertical", "vertical"), ("combine_broad_jump", "broad_jump"),
        ("combine_bench", "bench"), ("combine_ht", "ht_in"),
    ]:
        tol_key = pfr_field if pfr_field in TOLERANCES else pfr_field.replace("_in", "")
        tol = TOLERANCES.get(tol_key, 0.0)
        a = pd.to_numeric(matched[nfl_field], errors="coerce") if nfl_field in matched.columns else pd.Series(np.nan, index=matched.index)
        b = pd.to_numeric(matched[pfr_field], errors="coerce") if pfr_field in matched.columns else pd.Series(np.nan, index=matched.index)
        both = a.notna() & b.notna()
        if not both.any():
            print(f"{pfr_field:12s} {tol:>8.2f} {a.notna().sum():>6d} {b.notna().sum():>6d} {both.sum():>6d} {'-':>6s} {'-':>6s} {'-':>7s}")
            continue
        diff = (a[both] - b[both]).abs()
        ok = diff <= tol
        n_pass, n_fail = ok.sum(), (~ok).sum()
        total_checks += int(both.sum())
        total_pass += int(n_pass)
        pct = 100 * n_pass / both.sum()
        print(f"{pfr_field:12s} {tol:>8.2f} {a.notna().sum():>6d} {b.notna().sum():>6d} {both.sum():>6d} {n_pass:>6d} {n_fail:>6d} {pct:>6.1f}%")
        # Capture a few failure examples
        if n_fail and len(fail_examples) < 10:
            bad = matched.loc[both & ~ok].head(3)
            for _, r in bad.iterrows():
                fail_examples.append(
                    f"  {pfr_field}: {r.get('player_name_nfl', r.get('player_name'))!r} nfl={r[nfl_field]} pfr={r[pfr_field]}"
                )

    pass_rate = total_pass / total_checks if total_checks else 0.0
    print(f"\nOverall numeric pass rate: {total_pass}/{total_checks} = {pass_rate:.2%}  (threshold ≥ 99%)")

    if fail_examples:
        print("\nSample failures:")
        for line in fail_examples[:10]:
            print(line)

    # ── Final verdict ───────────────────────────────────────────────────────────
    print()
    pass_match = match_rate >= 0.95
    pass_field = pass_rate >= 0.99 if total_checks else False
    if pass_match and pass_field:
        print("PASS — PFR 2025 reconciles with nflverse 2025 within tolerance. Safe to merge 2026.")
        sys.exit(0)
    print("FAIL — investigate before merging PFR 2026.")
    sys.exit(1)


if __name__ == "__main__":
    main()
