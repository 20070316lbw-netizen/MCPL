import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.mcpl import leakage


def test_future_shock_changes_label_only():
    dates = pd.bdate_range("2024-12-02", periods=23)
    opening = pd.DataFrame({"A": 100.0}, index=dates)
    close = opening.copy()
    before = leakage._fields(opening, close)
    close.iloc[-1] = 110
    after = leakage._fields(opening, close)
    for feature in leakage.FEATURES:
        pd.testing.assert_frame_equal(before[feature].iloc[:-1], after[feature].iloc[:-1])
    assert np.isclose(after["target"].iloc[-2, 0], 100 * np.log(1.1))
    close.iloc[-2] = np.nan
    gap = leakage._fields(opening, close)
    assert np.isnan(gap["target"].iloc[-3, 0])
    assert np.isnan(gap["vol20"].iloc[-1, 0])


def test_purge_and_test_input():
    days = pd.to_datetime(["2024-12-30", "2024-12-31"])
    index = pd.MultiIndex.from_product([days, ["A"]], names=["date", "ticker"])
    panel = pd.DataFrame(1., columns=leakage.FEATURES, index=index)
    panel["label_date"] = pd.to_datetime(["2024-12-31", "2025-01-02"])
    panel["target"] = [5, 6]
    sliced = leakage._window(panel, "2024-01-01", "2024-12-31")
    assert sliced.index.tolist() == [index[0]]
    scale = StandardScaler().fit(panel[leakage.FEATURES])
    x = leakage._inputs(scale, panel)
    panel["target"] *= 100
    np.testing.assert_array_equal(x, leakage._inputs(scale, panel))
    assert not x[:, -1].any()
