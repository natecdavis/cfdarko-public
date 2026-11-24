import numpy as np

from src.models.availability import N_PICKS, AvailabilityModel


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    rank = rng.integers(1, 301, n)
    L = np.log(rank)
    logy = 0.4 + 0.85 * L + rng.normal(0, np.exp(-0.1 - 0.12 * L))
    pick = np.round(np.exp(logy)).clip(1)
    pick[pick > N_PICKS] = np.nan
    return rank, pick


def test_recovers_parameters_and_calibrates():
    rank, pick = _synthetic()
    m = AvailabilityModel().fit(rank, pick)
    assert abs(m.theta[1] - 0.85) < 0.15 and abs(m.theta[4] + 0.12) < 0.1
    y = np.nan_to_num(pick, nan=N_PICKS + 1)
    for n in (32, 100, 200):
        assert abs(m.survival(rank, n).mean() - (y > n).mean()) < 0.02


def test_p_available_bounds_and_monotone():
    rank, pick = _synthetic()
    m = AvailabilityModel().fit(rank, pick)
    r = np.array([1, 10, 50, 150, 300])
    p = m.p_available(r, 20, 52)
    assert np.all((p >= 0) & (p <= 1)) and np.all(np.diff(p) > 0)
    assert np.all(m.p_available(r, 20, 21) == 1)
    assert np.all(m.p_available(r, 20, 40) >= p)


def test_offboard_model_ranks_and_conditional():
    import pandas as pd

    from src.models.availability import OffboardAvailabilityModel, _off_features

    rng = np.random.default_rng(1)
    n = 6000
    d = pd.DataFrame({"cls": rng.integers(2015, 2021, n), "p": rng.random(n), "s": rng.random(n),
                      "pos": rng.choice(["WR", "OL", "K"], n), "inv": (rng.random(n) < 0.01).astype(float)})
    rank = d.groupby("cls")["p"].rank(ascending=False)
    drafted = rng.random(n) < 0.3 / rank
    pick = np.where(drafted, rng.integers(150, 263, n), np.nan)
    X, inv = _off_features(d, "cls", "p", "s", "pos", "inv")
    m = OffboardAvailabilityModel().fit(X, inv, pick)
    s, p = m.survival(X, inv, 262)
    assert abs(p.sum() - drafted.sum()) / drafted.sum() < 0.1
    assert np.corrcoef(p, -np.log(rank))[0, 1] > 0.5
    q = m.p_available(X, inv, 150, 200)
    assert np.all((q > 0) & (q <= 1)) and np.all(m.p_available(X, inv, 150, 151) == 1)
