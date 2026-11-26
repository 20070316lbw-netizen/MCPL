import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from model import make_model


def fit_and_evaluate(path, features, target):
    columns = ["date", "label_date", *features, target]
    if len(columns) != len(set(columns)):
        raise ValueError("Date columns, features and target must have distinct names.")

    panel = pd.read_parquet(path, columns=columns)
    date = pd.to_datetime(panel["date"], errors="raise")
    label_date = pd.to_datetime(panel["label_date"], errors="raise")
    if (label_date < date).any():
        raise ValueError("label_date cannot precede date.")

    start = pd.Timestamp("2019-01-01")
    cutoff = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2026-01-01")
    train = panel.loc[(date >= start) & (date < cutoff) & (label_date < cutoff)]
    test = panel.loc[(date >= cutoff) & (date < end) & (label_date < end)]
    train = train.dropna(subset=[*features, target])
    test = test.dropna(subset=[*features, target])
    if train.empty or test.empty:
        raise ValueError("The training and test windows both need complete rows.")

    x_train = train[features].to_numpy(dtype=np.float32)
    x_test = test[features].to_numpy(dtype=np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    model = make_model()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    baseline = np.full(len(test), y_train.mean(dtype=np.float64))

    print(f"Training rows: {len(train):,} | Test rows: {len(test):,}")
    for name, values in [("MLP", prediction), ("Train mean", baseline)]:
        mae = mean_absolute_error(y_test, values)
        rmse = root_mean_squared_error(y_test, values)
        print(f"{name:<10} MAE={mae:.6g}  RMSE={rmse:.6g}")


def main():
    parser = argparse.ArgumentParser(description="Train and test the MCPL baseline.")
    parser.add_argument("path", help="Parquet file with one row per stock and trading day")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    fit_and_evaluate(args.path, args.features, args.target)


if __name__ == "__main__":
    main()
