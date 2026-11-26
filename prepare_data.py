"""从版本化 Parquet 创建独立只读实验视图, 在仓库根目录运行"""

from hashlib import file_digest
import json
from pathlib import Path

import duckdb

from quant_research_lab.experiments import load_experiment_config


def main():
    cfg = load_experiment_config("leakage.yaml")
    root = Path("data") / f"quant-lab-data-{cfg.data.dataset_version}"
    manifest = json.loads((root / "manifest.json").read_text())
    with duckdb.connect(cfg.data.db_path) as db:
        db.begin()
        for table in cfg.data.tables:
            paths = []
            for entry in manifest["tables"][table]["files"]:
                if table == "prices" and not any(
                    f"year={year}/" in entry["path"]
                    for year in range(int(cfg.model.params["history_start"][:4]), cfg.data.end_date.year + 1)
                ):
                    continue
                path = (root / entry["path"]).resolve()
                with path.open("rb") as stream:
                    if file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                        raise ValueError(f"分区校验失败: {path.name}")
                paths.append(str(path))
            db.read_parquet(paths).create_view(table, replace=True)
        db.commit()


if __name__ == "__main__":
    main()
