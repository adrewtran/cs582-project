"""Extend the real product catalog with synthetic products for TabNet stress tests.

Reads data/crm/products.csv (read-only) and writes data/crm_synthetic/products.csv.
Also writes competitor_products.csv (rival products with recycled %, longevity, made-in-USA).
Synthetic rows are flagged so they are never mixed into final results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "crm" / "products.csv"
OUT = ROOT / "data" / "crm_synthetic" / "products.csv"

# Target industry per product, using the account `sector` labels (raw spelling, e.g. "technolgy").
# Every one of the 10 sectors gets at least one product so product x sector effects can be planted.
INDUSTRY = {
    "GTX Basic": "retail", "GTX Plus Basic": "services", "GTX Pro": "marketing",
    "GTX Plus Pro": "finance", "MG Special": "employment", "MG Advanced": "entertainment",
    "GTK 500": "telecommunications", "GTX Lite": "retail", "GTX Ultra": "technolgy",
    "MG Lite": "employment", "MG Elite": "software", "GTK 250": "telecommunications",
    "GTK 1000": "medical", "GTS Starter": "services", "GTS Standard": "marketing",
    "GTS Enterprise": "finance",
}

# Extra tiers inside existing series plus a new GTS series (price spread covers ~25 to ~52k).
NEW_PRODUCTS = [
    ("GTX Lite", "GTX", 250),
    ("GTX Ultra", "GTX", 9800),
    ("MG Lite", "MG", 25),
    ("MG Elite", "MG", 7200),
    ("GTK 250", "GTK", 13400),
    ("GTK 1000", "GTK", 52000),
    ("GTS Starter", "GTS", 1500),
    ("GTS Standard", "GTS", 3200),
    ("GTS Enterprise", "GTS", 15000),
]


# Environmental/sourcing factors compared against competitors: (recycled %, longevity years, made in USA).
# Invented values; higher series tiers skew longer-lived. Not real product facts.
FACTORS = {
    "GTX Basic": (25, 4, 1), "GTX Plus Basic": (30, 5, 1), "GTX Pro": (35, 7, 1), "GTX Plus Pro": (40, 8, 1),
    "MG Special": (15, 3, 0), "MG Advanced": (30, 6, 0), "GTK 500": (55, 12, 1), "GTX Lite": (20, 3, 1),
    "GTX Ultra": (50, 10, 1), "MG Lite": (10, 2, 0), "MG Elite": (45, 9, 0), "GTK 250": (45, 10, 1),
    "GTK 1000": (65, 15, 1), "GTS Starter": (20, 4, 0), "GTS Standard": (35, 6, 0), "GTS Enterprise": (60, 12, 1),
}
COMPETITOR_BRANDS = ["Nexora", "Vantrix", "Orbitek", "Kalmor", "Tession", "Brightline"]
COMPETITORS_PER_PRODUCT = 2
SEED = 582


def competitor_table(products: pd.DataFrame) -> pd.DataFrame:
    """Two rival products per own product, with factors scattered around ours (some better, some worse)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for _, p in products.iterrows():
        for k in range(COMPETITORS_PER_PRODUCT):
            brand = COMPETITOR_BRANDS[(len(rows)) % len(COMPETITOR_BRANDS)]
            rows.append({
                "competitor_product": f"{brand} {p['product']}-class",
                "competes_with": p["product"],
                "competitor_brand": brand,
                "sales_price": round(p["sales_price"] * rng.uniform(0.8, 1.25), 2),
                "recycled_pct": int(np.clip(p["recycled_pct"] + rng.normal(0, 20), 0, 100)),
                "longevity_years": round(float(np.clip(p["longevity_years"] * rng.lognormal(0, 0.3), 1, 25)), 1),
                "made_in_usa": int(rng.random() < 0.5),
            })
    return pd.DataFrame(rows)


def main() -> None:
    real = pd.read_csv(SRC).assign(synthetic=False)
    new = pd.DataFrame(NEW_PRODUCTS, columns=["product", "series", "sales_price"]).assign(synthetic=True)
    out = pd.concat([real, new], ignore_index=True)
    assert out["product"].is_unique, "duplicate product names"
    out["industry"] = out["product"].map(INDUSTRY)
    assert out["industry"].notna().all(), "product missing an industry"
    out[["recycled_pct", "longevity_years", "made_in_usa"]] = pd.DataFrame(
        out["product"].map(FACTORS).tolist(), index=out.index)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    competitors = competitor_table(out)
    competitors.to_csv(OUT.parent / "competitor_products.csv", index=False)
    print(out.to_string(index=False))
    print(competitors.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
