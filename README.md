# MCPL

股票日频面板上的 MLP 回归基线。模型使用两层隐藏层（32、16 个神经元）、ReLU 和线性输出，使用 Adam 按批训练。特征标准化参数只用训练集计算。

```bash
uv sync --locked
uv run --locked train.py data/panel.parquet --features momentum_20 volatility_20 --target return_next_day
```

输入 Parquet 每行对应一只股票的一个交易日。表里需要有 `date`（特征日期）、`label_date`（目标值最早可观测的日期）、指定的数值特征列和目标列。列名按实际数据替换。比如预测次日收益时，`label_date` 填次交易日。训练窗口为 2019–2024 年，测试窗口为 2025 年。每个窗口只使用在该窗口结束前已可观测的标签，有缺失特征或目标的行会被跳过。

运行后打印两个窗口的有效行数，以及 MLP 和训练集均值预测在同一个测试集上的 MAE、RMSE。

## 未来标签泄漏与补训实验

研究设定与结果见 [首轮实验报告](outputs/leakage/report.md)。工程规范从 QRL 的 db6551a 原样引入, 依赖锁定到同一提交。

在仓库根目录将 quant-lab-data-v1.0.0 放到 data/ 下, 其中需要 manifest.json, 2018 至 2025 年的 prices 分区及 universe 和 calendar 表。已有存储目录也可软链接到该位置。准备脚本按清单验证这些文件并创建独立 DuckDB 视图。

```bash
uv sync --locked
uv run --locked prepare_data.py
uv run --locked run_leakage.py
uv run --locked pytest
```

参数保存在 configs/leakage.yaml, 每次运行将更新 outputs/leakage/summary.json。
