"""Collect every ``baseline_metrics.csv`` under ``reports/`` into one table.

    python -m src.summary
"""

from __future__ import annotations

import pandas as pd

from src.baselines import REPORTS_DIR


def collect() -> pd.DataFrame:
    frames = []
    for path in sorted(REPORTS_DIR.glob("**/baseline_metrics.csv")):
        rel = path.parent.relative_to(REPORTS_DIR)
        table = pd.read_csv(path)
        table.insert(0, "split", "temporal" if rel.name == "temporal" else "random")
        table.insert(0, "dataset", rel.parts[0])
        frames.append(table)
    return pd.concat(frames, ignore_index=True).set_index(["dataset", "split", "model"])


def main() -> int:
    table = collect()
    table.to_csv(REPORTS_DIR / "summary.csv")
    print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
