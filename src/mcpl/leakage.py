"""历史标签泄漏与干净补训实验"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from model import make_model
from quant_research_lab.data.contracts import validate_panel
from quant_research_lab.data.loader import load_calendar, load_prices, load_universe_panel
from quant_research_lab.experiments import load_experiment_config
from quant_research_lab.stats import StatResult, adjust_pvalues

FEATURES = ["intraday", "mean5", "mean20", "vol20"]


def _fields(open_price, close_price):
    valid = (open_price > 0) & (close_price > 0)
    returns = 100 * np.log(close_price.where(valid) / open_price.where(valid))
    return {
        "intraday": returns,
        "mean5": returns.rolling(5).mean(),
        "mean20": returns.rolling(20).mean(),
        "vol20": returns.rolling(20).std(),
        "target": returns.shift(-1),
    }


def _load(cfg):
    db = Path(cfg.data.db_path)
    start = cfg.model.params["history_start"]
    end = cfg.data.end_date
    cal = load_calendar(start, end, db_path=db)
    members = load_universe_panel(cfg.data.start_date, end, columns=["cik"], db_path=db)
    tickers = members.index.unique("ticker").tolist()
    prices = load_prices(start, end, tickers=tickers, columns=["open", "close"],
                         sp500_only=False, db_path=db)
    fields = _fields(*(prices[col].unstack().reindex(cal) for col in ["open", "close"]))
    panel = pd.DataFrame({k: v.stack().reindex(members.index) for k, v in fields.items()})
    panel["label_date"] = panel.index.get_level_values("date").map(cal.to_series().shift(-1))
    return validate_panel(panel.dropna()), len(members)


def _window(panel, start, end):
    date = panel.index.get_level_values("date")
    return panel.loc[(date >= pd.Timestamp(start)) & (panel.label_date <= pd.Timestamp(end))]


def _inputs(scale, frame):
    x = scale.transform(frame[FEATURES]).astype("float32")
    return np.column_stack((x, np.zeros(len(x), dtype="float32")))


def _mlp(seed):
    model = make_model().steps[-1][1]
    return model.set_params(random_state=seed, shuffle=False)


def _run_seed(cfg, pre, tune, test, seed):
    p = cfg.model.params
    scale = StandardScaler().fit(pre[FEATURES])
    xp, xr, xt = [_inputs(scale, w) for w in [pre, tune, test]]
    yp, yr, yt = [w.target.to_numpy(dtype="float32") for w in [pre, tune, test]]
    leaky = xp.copy()
    leaky[:, -1] = yp
    models = {name: _mlp(seed) for name in ["clean", "exposed", "restart"]}
    rng = np.random.default_rng(seed)
    for _ in range(p["pretrain_epochs"]):
        order = rng.permutation(len(yp))
        for name, x in [("clean", xp), ("exposed", leaky)]:
            models[name].partial_fit(x[order], yp[order])
    pre_mae = {name: float(np.abs(models[name].predict(x) - yp).mean() * 100)
               for name, x in [("clean", xp), ("exposed", leaky)]}
    days = test.index.get_level_values("date")
    months = tune.label_date.dt.to_period("M")
    origin = pd.Period(p["recovery_start"], freq="M")
    daily = {}
    counts = {}
    seen = 0
    for month in range(max(p["checkpoints"]) + 1):
        if month:
            rows = np.flatnonzero(months == origin + month - 1)
            rows = rng.permutation(rows)
            for model in models.values():
                model.partial_fit(xr[rows], yr[rows])
            seen += len(rows)
        if month in p["checkpoints"]:
            counts[month] = seen
            for name, model in models.items():
                if name == "restart" and month == 0:
                    continue
                errors = 100 * np.abs(model.predict(xt) - yt)
                daily[month, name] = pd.Series(errors, index=days).groupby(level=0).mean()
    return pd.DataFrame(daily), pre_mae, counts


def main():
    cfg = load_experiment_config("leakage.yaml")
    p = cfg.model.params
    panel, eligible = _load(cfg)
    pre = _window(panel, cfg.data.start_date, pd.Timestamp(p["recovery_start"]) - pd.Timedelta(days=1))
    tune = _window(panel, p["recovery_start"], cfg.split.is_end_date)
    test = _window(panel, cfg.split.oos_start_date, cfg.data.end_date)
    by_seed = []
    diagnostics = {}
    with threadpool_limits(limits=1):
        for seed in p["seeds"]:
            daily, diagnostics[seed], counts = _run_seed(cfg, pre, tune, test, seed)
            by_seed.append(daily)
    mean = sum(by_seed) / len(by_seed)
    output = Path(cfg.output.dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for month in p["checkpoints"]:
        for other in ["clean", "restart"]:
            if (month, other) not in mean:
                continue
            gap = mean[month, "exposed"] - mean[month, other]
            fit = sm.OLS(gap, np.ones((len(gap), 1))).fit(
                cov_type="HAC", cov_kwds={"maxlags": cfg.audit.nw_lags})
            low, high = fit.conf_int(alpha=cfg.audit.alpha).iloc[0]
            delta, se = fit.params.iloc[0], fit.bse.iloc[0]
            critical = norm.isf(cfg.audit.alpha / (2 * cfg.audit.n_trials))
            bound = abs(delta) + critical * se
            stat = StatResult(
                name=f"{month}m_exposed_minus_{other}", estimate=delta, se=se,
                t_stat=fit.tvalues.iloc[0], p_value=fit.pvalues.iloc[0],
                ci_low=low, ci_high=high, n_obs=len(gap),
                meta={"month": month, "control": other, "simultaneous_abs_bound_bp": bound,
                      "equivalent": bool(bound < p["equivalence_bp"])},
                warnings=("单一测试年, HAC 近似推断, 随机种子先平均",))
            results.append(stat.to_dict())
    assert len(results) == cfg.audit.n_trials
    adjusted = adjust_pvalues([r["p_value"] for r in results], method=cfg.audit.p_adjust_method)
    for result, adj in zip(results, adjusted.p_adj, strict=True):
        result["p_holm"] = adj
    summary = {"configuration": cfg.model_dump(mode="json"), "eligible_stock_days": eligible,
               "rows": [len(w) for w in [pre, tune, test]], "test_days": len(mean),
               "test_tickers": test.index.unique("ticker").size,
               "zero_mae_bp": 100 * test.target.abs().groupby(level="date").mean().mean(),
               "pretrain_mae_bp": diagnostics, "recovery_rows": counts,
               "daily_equal_weight_mae_bp": {f"{m}m_{n}": v for (m,n),v in mean.mean().items()},
               "contrasts": results}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(summary["rows"], summary["daily_equal_weight_mae_bp"])


if __name__ == "__main__":
    main()
