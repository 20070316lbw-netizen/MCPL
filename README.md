# MCPL

研究股票日频预测的标签、损失函数和训练历史。PyTorch 实验位于 `src/mcpl/`。首轮 scikit-learn 实验的历史结果见 [报告](outputs/leakage/report.md)。

## 从原始数据跑一个小实验

在仓库根目录使用 Python 3.12 和 uv。把 quant-lab-data-v1.0.0 放在 `data/` 下, 需要 calendar、universe 和 2018–2025 年的 prices 分区。也可以给 `run_torch.py` 提供自有 Parquet, 列名和任务参数在 `configs/torch.toml` 中设置。

```bash
uv sync --locked
uv run --locked build_panel.py
uv run --locked --with 'torch==2.9.1' run_torch.py
uv run --locked --with 'torch==2.9.1' pytest -q
```

默认在 CPU 上运行, 用最近 21 个有效信号日做预训练, 在第 0、1、5、10、20 个干净补训日检查, 2025 年固定测试。输出为 `runs/torch/scores.csv` 与本次参数快照。可将 windows 改为 [21, 63, 252, 0], 其中 0 表示预训练窗口内的全部历史。每轮训练按日随机排列横截面, 每个批次含同日股票。由于初始化、批次和优化器实现发生变化, 新结果应单列为 Torch 实验。

## 组装方式

| 部件 | 接口 | 用途 |
|---|---|---|
| 面板和标签 | build_panel, select_window, rank_labels | PIT 成分、连续交易日特征、跨窗口标签剔除、截面平均名次 |
| 网络 | MLP(n_features, hidden) | 可调隐藏层, 每只股票输出一个标量 |
| 损失 | torch.nn.MSELoss、HuberLoss、PairwiseRankLoss | 回归误差或同一天标签不同的股票对排序 |
| 训练 | Trainer(model, loss, optimizer) | 可独立替换三项, 多次调用 fit_epoch 保留优化器状态 |
| 对照实验 | run(panel, config) | 组装干净组、泄漏组和重新初始化组, 汇总 MAE 和逐日 Rank IC |

例如, 从仓库根目录在 Python 中构建一个用自定义排序损失的训练器:

```python
import torch
from src.mcpl.model import MLP
from src.mcpl.losses import PairwiseRankLoss
from src.mcpl.training import Trainer

model = MLP(n_features=5, hidden=(16, 8))
trainer = Trainer(model, PairwiseRankLoss(), torch.optim.Adam(model.parameters(), lr=0.001))
```

自定义损失接收同日的预测分数和标签两个一维张量, 返回可求导标量。排名标签映射至 [0, 1], 单只股票日剔除, 并列取平均名次。Rank IC 为每日 Spearman 相关的有效日均值。配对损失优化次序, 分数的绝对水平没有校准, 此时优先读 Rank IC。对照组共用训练期特征尺度, 只在泄漏分支预训练时把最后一列设为真实标签, 其他阶段该列置零。

## 首轮实验复现

```bash
uv run --locked prepare_data.py
uv run --locked run_leakage.py
```

该入口沿用首轮 sklearn 设置和锁定的 QRL 数据接口。旧版自有面板基线仍可通过 `train.py` 运行。
