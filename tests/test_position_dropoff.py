import itertools

import numpy as np
import pandas as pd

from src.models.position_dropoff import (expected_best, _expected_best_excluding_each, dropoff,
                                         view_position)


def brute(v, p):
    """E[max available] by enumerating every availability pattern."""
    tot = 0.0
    for pat in itertools.product([0, 1], repeat=len(v)):
        pr = np.prod([pi if a else 1 - pi for a, pi in zip(pat, p)])
        avail = [vi for a, vi in zip(pat, v) if a]
        tot += pr * (max(avail) if avail else 0.0)
    return tot


def test_expected_best_matches_enumeration():
    rng = np.random.default_rng(1)
    v, p = rng.random(6), rng.random(6)
    assert np.isclose(expected_best(v, p), brute(v, p))


def test_excluding_each_matches_enumeration():
    rng = np.random.default_rng(2)
    v, p = rng.random(6), rng.random(6)
    p[2] = 1.0                      # exercise the certain-to-be-there branch
    ex = _expected_best_excluding_each(v, p)
    for k in range(6):
        m = np.arange(6) != k
        assert np.isclose(ex[k], brute(v[m], p[m]))


def test_dropoff_table_and_view_position():
    pool = pd.DataFrame({"player_name": list("abcd"), "view_pos": ["OT", "OT", "WR", "WR"],
                         "v_now": [0.3, 0.2, 0.25, 0.1], "v_next": [0.28, 0.18, 0.23, 0.09],
                         "p_next": [0.1, 0.9, 0.5, 1.0]})
    out, T = dropoff(pool)
    ot = T.set_index("pos").loc["OT"]
    assert ot["best_now"] == "a" and np.isclose(ot["exp_next"], expected_best(np.array([0.28, 0.18]),
                                                                             np.array([0.1, 0.9])))
    assert np.isclose(out.loc[0, "pos_wait_cost"], 0.3 - ot["exp_next"])
    vp = view_position(pd.Series(["OL", "OL", "OL", "DB"]), pd.Series(["OT", "OG", "OL", "CB"]))
    assert vp.tolist() == ["OT", "IOL", "OL", "DB"]
