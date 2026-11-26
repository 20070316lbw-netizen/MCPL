from pathlib import Path
from src.mcpl.panel import build_panel


if __name__ == "__main__":
    output = Path("runs/panel.parquet")
    output.parent.mkdir(exist_ok=True)
    build_panel("data/quant-lab-data-v1.0.0").to_parquet(output, index=False)
    print(output)
