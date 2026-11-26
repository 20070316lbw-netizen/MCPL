import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .losses import PairwiseRankLoss
from .model import MLP
from .panel import rank_ic, rank_labels, select_window
from .training import Trainer, day_batches


def make_trainer(width, config, seed):
    torch.manual_seed(seed)
    model = MLP(width, config["hidden"])
    losses = {"mse": nn.MSELoss, "huber": nn.HuberLoss, "pairwise": PairwiseRankLoss}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    return Trainer(model, losses[config["loss"]](), optimizer)


def inputs(scaler, frame, features):
    clean = torch.tensor(scaler.transform(frame[features]), dtype=torch.float32)
    return torch.cat([clean, torch.zeros(len(frame), 1)], dim=1)


def score(trainer, x, frame, target):
    predictions = trainer.predict(x).numpy()
    ic = rank_ic(frame, predictions, target)
    error = pd.Series(abs(predictions - frame["label"].to_numpy()), index=frame.index)
    return {"mae": error.groupby(frame.date).mean().mean(), "rank_ic": ic.mean(),
            "valid_ic_days": ic.count(), "test_days": frame.date.nunique()}


def run_window(pre, recovery, test, config, seed):
    features = config["features"]
    scaler = StandardScaler().fit(pre[features])
    xp, xr, xt = [inputs(scaler, frame, features) for frame in (pre, recovery, test)]
    y, yr = [torch.tensor(frame.label.to_numpy(), dtype=torch.float32) for frame in (pre, recovery)]
    leaked = xp.clone()
    leaked[:, -1] = y
    trainers = {name: make_trainer(xp.shape[1], config, seed)
                for name in ("clean", "exposed", "restart")}
    rng = np.random.default_rng(seed)
    for _ in range(config["epochs"]):
        batches = list(day_batches(pre.date, rng))
        trainers["clean"].fit_epoch(xp, y, batches)
        trainers["exposed"].fit_epoch(leaked, y, batches)

    days = list(recovery.groupby("label_date", sort=True).indices.values())
    checkpoints = config["checkpoints"]
    if min(checkpoints) < 0 or max(checkpoints) > len(days):
        raise ValueError("checkpoint outside the recovery window")
    seen = 0
    for step in range(max(checkpoints) + 1):
        if step:
            rows = days[step - 1]
            for trainer in trainers.values():
                trainer.fit_epoch(xr, yr, [rows])
            seen += len(rows)
        if step in checkpoints:
            for name, trainer in trainers.items():
                if step == 0 and name == "restart":
                    continue
                yield {"seed": seed, "pre_days": pre.date.nunique(), "pre_rows": len(pre),
                       "day": step, "recovery_rows": seen, "model": name,
                       "target": config["label"], "loss": config["loss"],
                       **score(trainer, xt, test, config["target"])}


def run(panel, config):
    target = config["target"]
    features = config["features"]
    if len(set(features + [target, "date", "ticker", "label_date", "label"])) != len(features) + 5:
        raise ValueError("feature, target and reserved column names must be distinct")
    panel = panel.dropna(subset=["date", "ticker", "label_date", target, *features]).copy()
    for col in ("date", "label_date"):
        panel[col] = pd.to_datetime(panel[col])
    if panel.duplicated(["date", "ticker"]).any() or (panel.label_date <= panel.date).any():
        raise ValueError("expected unique ticker-days with a future label")
    if panel.groupby("date").label_date.nunique().gt(1).any():
        raise ValueError("inconsistent prediction horizon within date")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    if config["label"] == "rank":
        panel["label"] = rank_labels(panel, target)
    elif config["label"] == "return":
        panel["label"] = panel[target]
    else:
        raise ValueError("label must be return or rank")
    panel = panel.dropna(subset=["label"])
    pre, recovery, test = [select_window(panel, *config[key]) for key in ("pretrain", "recovery", "test")]
    if any(w.empty for w in (pre, recovery, test)):
        raise ValueError("each experiment window needs complete rows")
    if pre.label_date.max() >= recovery.date.min() or recovery.label_date.max() >= test.date.min():
        raise ValueError("overlapping experiment windows")
    dates = pre.date.drop_duplicates().to_numpy()
    records = []
    for length in config["windows"]:
        if length < 0 or length > len(dates):
            raise ValueError("pretraining window exceeds available dates")
        sample = pre.loc[pre.date >= dates[-length]] if length else pre
        for seed in config["seeds"]:
            records.extend(run_window(sample, recovery, test, config, seed))
    return pd.DataFrame(records)
