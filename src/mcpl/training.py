import torch


class Trainer:
    """持有模型、损失和优化器, 多次调用 fit_epoch 可接续训练"""

    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def fit_epoch(self, x, y, batches):
        self.model.train()
        for rows in batches:
            self.optimizer.zero_grad(set_to_none=True)
            value = self.loss(self.model(x[rows]), y[rows])
            value.backward()
            self.optimizer.step()

    @torch.no_grad()
    def predict(self, x, batch_size=4096):
        self.model.eval()
        return torch.cat([self.model(batch) for batch in x.split(batch_size)])


def day_batches(dates, rng):
    """每个 batch 只含同一个信号日的股票, 适用于回归和配对排序"""
    groups = list(dates.groupby(dates, sort=True).indices.values())
    return (groups[index] for index in rng.permutation(len(groups)))
