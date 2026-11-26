from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["intraday", "mean5", "mean20", "vol20"]


def build_panel(root, history_start="2018-11-01", start="2019-01-01", end="2025-12-31"):
    """在完整交易日历构造特征与次日标签, 按信号日匹配历史成分快照"""
    root = Path(root) / "data"
    calendar = pd.read_parquet(root / "calendar/data.parquet", columns=["date"])
    calendar["date"] = pd.to_datetime(calendar.date).astype("datetime64[ns]")
    calendar = calendar.loc[calendar.date.between(history_start, end)].sort_values("date")
    snapshots = pd.read_parquet(root / "universe/data.parquet", columns=["snapshot_date", "ticker"])
    snapshots["snapshot_date"] = pd.to_datetime(snapshots.snapshot_date).astype("datetime64[ns]")
    latest = pd.merge_asof(calendar.loc[calendar.date >= start],
                           snapshots[["snapshot_date"]].drop_duplicates().sort_values("snapshot_date"),
                           left_on="date", right_on="snapshot_date")
    members = latest.merge(snapshots, on="snapshot_date").set_index(["date", "ticker"]).sort_index()
    if not members.index.is_unique:
        raise ValueError("duplicate constituents on signal date")
    tickers = members.index.unique("ticker").tolist()
    first, last = date.fromisoformat(history_start), date.fromisoformat(end)
    prices = pd.concat([
        pd.read_parquet(root / "prices" / f"year={year}", columns=["date", "ticker", "open", "close"],
                        filters=[("ticker", "in", tickers), ("date", ">=", first), ("date", "<=", last)])
        for year in range(first.year, last.year + 1)
    ])
    prices["date"] = pd.to_datetime(prices.date)
    valid = (prices.open > 0) & (prices.close > 0)
    returns = (100 * np.log(prices.close.where(valid) / prices.open.where(valid)))
    dates = pd.DatetimeIndex(calendar.date)
    wide = prices.assign(intraday=returns).pivot(index="date", columns="ticker", values="intraday").reindex(dates)
    fields = [wide, wide.rolling(5).mean(), wide.rolling(20).mean(), wide.rolling(20).std(), wide.shift(-1)]
    panel = pd.DataFrame({key: field.stack().reindex(members.index)
                          for key, field in zip([*FEATURES, "target"], fields, strict=True)})
    panel["label_date"] = panel.index.get_level_values("date").map(dates.to_series().shift(-1))
    return panel.dropna().reset_index()


def select_window(panel, start, end):
    return panel.loc[(panel.date >= start) & (panel.label_date <= end)]


def rank_labels(frame, target):
    groups = frame.groupby("date")[target]
    return (groups.rank() - 1) / (groups.transform("count") - 1).replace(0, np.nan)


def rank_ic(frame, scores, target):
    ranked = frame.assign(score=scores).groupby("date")[["score", target]].rank()
    return ranked.score.groupby(frame.date).corr(ranked[target])
