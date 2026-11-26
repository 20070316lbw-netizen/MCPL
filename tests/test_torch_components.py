import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from src.mcpl.experiment import run
from src.mcpl.losses import PairwiseRankLoss
from src.mcpl.model import MLP
from src.mcpl.training import Trainer


def test_pair_order_ties_and_gradients():
    loss = PairwiseRankLoss()
    score = torch.tensor([0., 1.], requires_grad=True)
    assert loss(score, score.detach()) < loss(-score, score.detach())
    loss(score, torch.ones(2)).backward()
    assert torch.equal(score.grad, torch.zeros(2))


def test_custom_loss_and_optimizer_update_weights():
    model = MLP(2, hidden=())
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].bias.zero_()
    trainer = Trainer(model, torch.nn.L1Loss(), torch.optim.SGD(model.parameters(), lr=0.1))
    x = torch.ones(4, 2)
    trainer.fit_epoch(x, torch.ones(4), [torch.arange(4)])
    assert trainer.predict(x).shape == (4,)
    assert (trainer.predict(x) > 0).all()


def test_recovery_experiment_end_to_end():
    dates = pd.bdate_range("2023-12-18", periods=15)
    panel = pd.DataFrame([
        dict(date=day, label_date=dates[i+1], ticker=str(stock), x=stock, y=(stock+i)%3)
        for i, day in enumerate(dates[:-1]) for stock in range(3)
    ])
    cfg = dict(features=["x"], target="y", label="rank", loss="pairwise",
               hidden=[4], learning_rate=.01, seeds=[1], epochs=1, windows=[2],
               checkpoints=[0, 1], pretrain=[dates[0], dates[4]],
               recovery=[dates[5], dates[8]], test=[dates[9], dates[-1]])
    result = run(panel, cfg)
    assert len(result) == 5
    assert set(result.pre_rows) == {6}
    assert set(result.recovery_rows) == {0, 3}
