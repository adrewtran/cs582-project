"""Build one dataset, write its processed table, print the cleaning report.

    python -m src.datasets leads
"""

from __future__ import annotations

import argparse
import json

from src.data import DATA_DIR
from src.datasets import LOADERS, load


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(LOADERS))
    args = parser.parse_args(argv)

    dataset = load(args.dataset)
    out_dir = DATA_DIR / args.dataset / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_csv(out_dir / "labeled.csv", index=False)
    for key, value in dataset.extra.items():
        if hasattr(value, "to_csv"):
            value.to_csv(out_dir / f"{key}.csv", index=False)

    print(json.dumps(dataset.report, indent=2, default=str))
    print(f"\n{len(dataset.categorical)} categorical + {len(dataset.numeric)} numeric features")
    print(f"Wrote {out_dir / 'labeled.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
