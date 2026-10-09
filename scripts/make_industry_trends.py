"""Simulated 5-year industry trend table for TabNet stress tests.

Writes data/crm_simulated/industry_trends.csv: one row per (industry, year) for 2011-2016
with annual growth_pct and demand_index (2011 = 100). Industries are the account `sector`
labels. Deals run 2016-10..2017-12, so a deal in year Y may only use years Y-5..Y-1
(`trailing_features`), which keeps the feature known before the deal is engaged.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "crm_simulated" / "industry_trends.csv"
YEARS = list(range(2011, 2017))
SEED = 582

# (base annual growth %, shape) per industry; shapes plant different trend patterns.
ARCHETYPES = {
    "technolgy": (12, "rising"),
    "software": (15, "accelerating"),
    "medical": (7, "steady"),
    "finance": (4, "steady"),
    "retail": (-1, "declining"),
    "marketing": (5, "cyclical"),
    "entertainment": (6, "cyclical"),
    "telecommunications": (2, "flat"),
    "services": (3, "flat"),
    "employment": (-3, "shock"),  # dip in 2013-2014, partial recovery
}


def growth_path(base: float, shape: str, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(len(YEARS))
    if shape == "accelerating":
        g = base * (0.5 + 0.25 * t)
    elif shape == "declining":
        g = base - 1.0 * t
    elif shape == "cyclical":
        g = base + 4 * np.sin(t * 1.3)
    elif shape == "shock":
        g = base + np.where((t >= 2) & (t <= 3), -10, np.where(t >= 4, 8, 0))
    elif shape == "rising":
        g = base + 0.5 * t
    else:  # steady / flat
        g = np.full(len(YEARS), float(base))
    return np.round(g + rng.normal(0, 1.0, len(YEARS)), 2)


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for industry, (base, shape) in ARCHETYPES.items():
        growth = growth_path(base, shape, rng)
        index = 100 * np.cumprod(1 + growth / 100)
        for y, g, i in zip(YEARS, growth, index):
            rows.append((industry, y, g, round(float(i), 2)))
    return pd.DataFrame(rows, columns=["industry", "year", "growth_pct", "demand_index"])


def trailing_features(trends: pd.DataFrame, deal_year: int) -> pd.DataFrame:
    """Per-industry trend over the 5 full years before `deal_year` (no look-ahead)."""
    w = trends[trends["year"].between(deal_year - 5, deal_year - 1)]
    out = w.groupby("industry").apply(
        lambda d: pd.Series({
            "trend_5y_avg_growth": d["growth_pct"].mean(),
            "trend_5y_slope": np.polyfit(d["year"], d["growth_pct"], 1)[0] if len(d) > 1 else 0.0,
            "trend_last_growth": d.sort_values("year")["growth_pct"].iloc[-1],
        }),
        include_groups=False,
    )
    return out.reset_index()


def main() -> None:
    trends = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    trends.to_csv(OUT, index=False)
    print(trends.pivot(index="industry", columns="year", values="growth_pct").to_string())
    print(trailing_features(trends, 2017).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
