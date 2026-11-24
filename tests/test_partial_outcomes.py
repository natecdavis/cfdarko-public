import numpy as np
import pandas as pd

from src.models.partial_outcomes import POS, project_partial


def _frame(seed=0, n_per=300, years=range(2014, 2026)):
    rng = np.random.default_rng(seed)
    rows = []
    for dy in years:
        r = rng.integers(1, 301, n_per)
        talent = np.exp(-np.log(r) / 2) * rng.lognormal(0, 0.3, n_per)
        v = np.column_stack([talent * f for f in (0.6, 0.9, 1.0, 1.0)]) * 0.03
        d = pd.DataFrame({"hasan_rank": r, "pos_group": rng.choice(POS, n_per), "draft_year": dy,
                          "last_college_season_year": dy - 1})
        for k in range(4):
            played = 2026 - dy
            d[f"v{k + 1}"] = np.where(k < played, v[:, k], np.nan)
        d["true"] = v.sum(1)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_projects_recent_classes_only_and_tracks_truth():
    b = _frame()
    rows, diag = project_partial(b, 2026)
    assert set(diag) == {2023, 2024, 2025}
    assert set(rows["last_college_season_year"]) == {2022, 2023, 2024}
    # rows come out by seasons played (k = 1, 2, 3 → newest class first)
    recent = b[b["draft_year"] >= 2023].dropna(subset=["v1"]).sort_values("draft_year", ascending=False,
                                                                          kind="mergesort")
    assert len(rows) == len(recent)
    m = rows["rv"].to_numpy()
    t = recent["true"].to_numpy()
    assert np.corrcoef(m, t)[0, 1] > 0.8
    assert abs(m.mean() / t.mean() - 1) < 0.1
    played = recent[["v1", "v2", "v3"]].sum(axis=1, min_count=1).to_numpy()
    assert np.all(m >= played - 1e-12)
