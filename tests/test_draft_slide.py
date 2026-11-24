import numpy as np

from src.models.draft_slide import value_at_pick, RATIO_MAX, SLIDE_GAMMA


def test_unchanged_at_board_rank_and_off_board():
    v = value_at_pick([0.1, 0.2], [20, np.nan], [20, 150])
    assert np.allclose(v, [0.1, 0.2])


def test_slide_lowers_rise_unchanged():
    slid, rose = value_at_pick([0.1, 0.1], [20, 20], [40, 10])
    assert np.isclose(slid, 0.1 * 2 ** SLIDE_GAMMA) and slid < 0.1
    assert np.isclose(rose, 0.1)


def test_ratio_capped():
    far = value_at_pick(0.1, 1, 200)
    assert np.isclose(far, 0.1 * RATIO_MAX ** SLIDE_GAMMA)
