"""Tests for the isotonic stability guard in step 8c.

Context: n_positive alone used to decide isotonic-vs-Platt, with no check on
whether the fitted map was well-behaved. EDGE/impact_apy had 240 positive OOF
rows (well over ISO_THRESHOLD=100) but produced a map that jumped
0.06922 -> 0.16100 across a 0.4% change in input. The whole current class
landed just below that edge and scored 0.036 — under LB and DB, for a premium
position. The guard rejects such a fit in favour of monotone-linear
Platt-Ridge, which cannot cliff.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent.parent))


def _c8():
    if "p8c_stability" in sys.modules:
        return sys.modules["p8c_stability"]
    mod_path = Path(__file__).parent.parent / "pipeline" / "08c_calibrate_draft_scores.py"
    spec = importlib.util.spec_from_file_location("p8c_stability", mod_path)
    mod = types.ModuleType("p8c_stability")
    mod.__file__ = str(mod_path)
    sys.modules["p8c_stability"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cliff_data(n=400, seed=0):
    """Reproduces the real EDGE/impact_apy shape: a top plateau supported by a
    SINGLE observation, sitting far above the level below it, so the map's
    final step is a cliff (real EDGE: 0.06922 -> 0.16100, a 2.33x jump).
    """
    rng = np.random.default_rng(seed)
    preds = np.sort(rng.uniform(0.0002, 0.033, n))
    targets = np.zeros(n)
    targets[-40:] = rng.uniform(0.02, 0.07, 40)
    targets[-1] = 0.161            # lone outlier defines the whole top level
    return preds, targets


def _smooth_data(n=400, seed=0):
    """Targets that rise steadily with predictions — no cliff."""
    rng = np.random.default_rng(seed)
    preds = np.sort(rng.uniform(0.0002, 0.033, n))
    targets = np.clip(preds * 2.0 + rng.normal(0, 0.002, n), 0, None)
    return preds, targets


class TestTopStepRatio:
    def test_flat_top_reports_one(self):
        """A map that plateaus at the top is healthy (DB and DL both do)."""
        mod = _c8()
        p, t = _smooth_data()
        cal = IsotonicRegression(y_min=0.0, y_max=float(t.max()),
                                 out_of_bounds="clip").fit(p, t)
        assert mod._isotonic_top_step_ratio(cal) == pytest.approx(1.0, abs=0.5)

    def test_cliff_is_detected(self):
        mod = _c8()
        p, t = _cliff_data()
        cal = IsotonicRegression(y_min=0.0, y_max=float(t.max()),
                                 out_of_bounds="clip").fit(p, t)
        assert mod._isotonic_top_step_ratio(cal) > mod.MAX_TOP_STEP_RATIO

    def test_degenerate_map_does_not_crash(self):
        """A map with fewer than two positive levels has no 'top step'."""
        mod = _c8()
        p = np.linspace(0.001, 0.03, 50)
        t = np.zeros(50)
        cal = IsotonicRegression(y_min=0.0, y_max=1.0,
                                 out_of_bounds="clip").fit(p, t)
        assert mod._isotonic_top_step_ratio(cal) == 1.0


class TestFallbackBehaviour:
    def test_cliff_no_longer_swaps_method(self):
        """Pins the 2026-07-27 REVERT. Falling back to Platt-Ridge was worse
        than the cliff: fitting a line through a ~93%-zero target flattened the
        range (QB calibrated max 0.140 -> 0.028, LB 0.097 -> 0.010), dropping
        QB from 1st to 8th on the board. Detection is kept; the swap is not."""
        mod = _c8()
        p, t = _cliff_data()
        cal, method = mod._fit_calibrator(p, t, "isotonic", 0.0, float(t.max()))
        assert method == "isotonic"
        assert isinstance(cal, IsotonicRegression)

    def test_smooth_fit_keeps_isotonic(self):
        """The guard must not fire on well-behaved maps — otherwise it silently
        downgrades every head."""
        mod = _c8()
        p, t = _smooth_data()
        cal, method = mod._fit_calibrator(p, t, "isotonic", 0.0, float(t.max()))
        assert method == "isotonic"
        assert isinstance(cal, IsotonicRegression)

    def test_effective_method_is_returned_not_the_requested_one(self):
        """_apply dispatches on the method string and Ridge needs 2-D input, so
        returning the requested method after a swap would crash inference."""
        mod = _c8()
        p, t = _cliff_data()
        cal, method = mod._fit_calibrator(p, t, "isotonic", 0.0, float(t.max()))
        out = mod._apply(method, cal, p, 0.0, float(t.max()))
        assert out.shape == p.shape
        assert np.isfinite(out).all()

    def test_explicit_platt_request_is_unchanged(self):
        mod = _c8()
        p, t = _smooth_data()
        cal, method = mod._fit_calibrator(p, t, "platt_ridge", 0.0, float(t.max()))
        assert method == "platt_ridge" and isinstance(cal, Ridge)

    def test_identity_is_unchanged(self):
        mod = _c8()
        p, t = _smooth_data()
        cal, method = mod._fit_calibrator(p, t, "identity", 0.0, 1.0)
        assert method == "identity" and cal is None


class TestCliffIsStillDetectedAndReported:
    """The cliff is real and remains UNFIXED — it is detected and logged so it
    stays visible, but no remedy is applied automatically."""

    def test_the_cliff_is_still_present_in_the_output(self):
        """Documents the open problem: two inputs 0.6% apart still produce
        outputs differing by a multiple. Fixing this properly most likely
        belongs in presentation (within-position ranks — isotonic is monotone,
        so within-position ordering is unaffected), not in the calibrator."""
        mod = _c8()
        p, t = _cliff_data()
        y_max = float(t.max())
        cal, method = mod._fit_calibrator(p, t, "isotonic", 0.0, y_max)
        xs = np.unique(np.round(cal.X_thresholds_, 6))
        edge = xs[-1]
        lo = mod._apply(method, cal, np.array([edge * 0.994]), 0.0, y_max)[0]
        hi = mod._apply(method, cal, np.array([edge * 1.006]), 0.0, y_max)[0]
        assert hi > 1.75 * max(lo, 1e-9)

    def test_detection_still_works(self):
        mod = _c8()
        p, t = _cliff_data()
        cal, _ = mod._fit_calibrator(p, t, "isotonic", 0.0, float(t.max()))
        assert mod._isotonic_top_step_ratio(cal) > mod.MAX_TOP_STEP_RATIO


class TestConfig:
    def test_threshold_is_above_one(self):
        """<=1.0 would reject every map, including flat ones."""
        mod = _c8()
        assert mod.MAX_TOP_STEP_RATIO > 1.0

    def test_count_thresholds_untouched(self):
        """The guard is a SHAPE criterion; it must not have been implemented by
        quietly moving the count thresholds."""
        mod = _c8()
        assert mod.ISO_THRESHOLD == 100
        assert mod.PLATT_THRESHOLD == 50


class TestKnownLimitation:
    """The guard inspects only the FINAL step of the map.

    A cliff one level down is missed. This was found while writing these tests:
    two adjacent outliers (0.160 and 0.161) produce levels [..., 0.050, 0.160,
    0.161], where the real cliff is 0.050 -> 0.160 but the final step is only
    1.006x. On the 43 production calibrators as of 2026-07-26 the final-step
    check happens to catch the right heads (EDGE 2.33 and OL 2.06, with a clean
    gap to LB at 1.73), so the narrow check is deliberate — a broader
    "largest step anywhere" criterion would also fire on heads whose big step
    may be legitimate (DL_role spans a bimodal [0,1] target). Pinned so the
    limitation is visible rather than rediscovered.
    """

    def test_cliff_one_level_down_is_not_caught(self):
        mod = _c8()
        rng = np.random.default_rng(0)
        n = 400
        p = np.sort(rng.uniform(0.0002, 0.033, n))
        t = np.zeros(n)
        t[-40:] = rng.uniform(0.02, 0.07, 40)
        t[-2:] = [0.160, 0.161]        # TWO outliers, not one
        cal = IsotonicRegression(y_min=0.0, y_max=float(t.max()),
                                 out_of_bounds="clip").fit(p, t)
        ys = np.unique(cal.y_thresholds_)
        ys = ys[ys > 0]
        assert ys[-1] / ys[-2] < mod.MAX_TOP_STEP_RATIO      # not caught
        assert ys[-2] / ys[-3] > mod.MAX_TOP_STEP_RATIO      # the real cliff
