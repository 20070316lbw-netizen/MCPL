from itertools import pairwise

from torch import nn


class MLP(nn.Module):
    """隐藏层宽度由调用方指定, 每只股票输出一个分数"""

    def __init__(self, n_features, hidden=(32, 16)):
        super().__init__()
        layers = []
        for n_in, n_out in pairwise((n_features, *hidden)):
            layers.extend([nn.Linear(n_in, n_out), nn.ReLU()])
        layers.append(nn.Linear(hidden[-1] if hidden else n_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)
