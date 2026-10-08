"""Rank products for future investment from the synthetic CRM data.

Reads data/crm_synthetic/ (sales_pipeline, products, product_rd, industry_trends) and writes
product_investment.csv there. Uses only observed outcomes, never ground_truth.csv.
Synthetic data and a heuristic score: an illustration of the analysis, not a business finding.

Columns: won deals and revenue, gross profit (revenue - unit_cost per won deal), total R&D,
R&D ROI ((gross profit - R&D) / R&D), win rate, margin, open pipeline expected value
(open deals x list price x observed win rate), and the target industry's latest growth.
``invest_score`` averages the percentile ranks of ROI, win rate, margin, pipeline-to-R&D and
industry growth (equal weights).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_industry_trends import trailing_features  # noqa: E402

SYN = Path(__file__).resolve().parents[1] / "data" / "crm_synthetic"
SCORE_PARTS = ["rd_roi", "win_rate", "gross_margin", "pipeline_to_rd", "industry_growth"]


def main() -> None:
    pipe = pd.read_csv(SYN / "sales_pipeline.csv")
    products = pd.read_csv(SYN / "products.csv").set_index("product")
    rd = pd.read_csv(SYN / "product_rd.csv").groupby("product")["rd_expense"].sum()
    trends = trailing_features(pd.read_csv(SYN / "industry_trends.csv"), 2017).set_index("industry")

    g = pipe.groupby("product")
    won = pipe[pipe["deal_stage"] == "Won"].groupby("product")
    closed = pipe["deal_stage"].isin(["Won", "Lost"])
    r = pd.DataFrame({
        "won_deals": won.size(),
        "revenue": won["close_value"].sum(),
        "win_rate": pipe[closed].groupby("product")["deal_stage"].apply(lambda s: (s == "Won").mean()),
        "open_deals": pipe[pipe["deal_stage"].isin(["Engaging", "Prospecting"])].groupby("product").size(),
    }).reindex(products.index).fillna(0)
    r["gross_profit"] = r["revenue"] - r["won_deals"] * products["unit_cost"]
    r["gross_margin"] = r["gross_profit"] / r["revenue"].where(r["revenue"] > 0)
    r["rd_expense"] = rd
    r["rd_roi"] = (r["gross_profit"] - r["rd_expense"]) / r["rd_expense"]
    r["pipeline_ev"] = r["open_deals"] * products["sales_price"] * r["win_rate"]
    r["pipeline_to_rd"] = r["pipeline_ev"] / r["rd_expense"]
    r["industry"] = products["industry"]
    r["industry_growth"] = r["industry"].map(trends["trend_last_growth"])
    r["invest_score"] = r[SCORE_PARTS].rank(pct=True).mean(axis=1)
    r = r.sort_values("invest_score", ascending=False).round(3)
    r.to_csv(SYN / "product_investment.csv", index_label="product")
    show = ["won_deals", "revenue", "gross_margin", "rd_expense", "rd_roi", "win_rate",
            "pipeline_ev", "industry", "industry_growth", "invest_score"]
    print(r[show].to_string())


if __name__ == "__main__":
    main()
