import json
import tomllib
from pathlib import Path

import pandas as pd

from src.mcpl.experiment import run


if __name__ == "__main__":
    with Path("configs/torch.toml").open("rb") as stream:
        config = tomllib.load(stream)
    output = Path(config["output"])
    result = run(pd.read_parquet(config["panel"]), config)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "scores.csv", index=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(result.to_string(index=False))
