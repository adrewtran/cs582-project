# Simulated CRM data: how it is generated

Simulated deals are used only to test whether the models (LR, RF, MLP, TabNet) can recover
structure that was deliberately planted. They are **not** evidence about real sales behaviour,
and results on them must never be mixed into `reports/crm/final/`.

Method: **parametric simulation from a known logistic model**. The win probability of every
deal is written down explicitly and outcomes are drawn from it. No GAN, CTGAN, SMOTE or copula is
used, so the true probability is known for every row (the "oracle" in the results).

## 1. Information sources

### Taken from the real CRM data (`data/crm/`, read-only)
- The 85 accounts (sector, revenue, employees) and 30 sales agents (manager, regional office).
- The 7 real products with series and list price.
- How often each real product is sold in the real pipeline (product sampling weights).
- Calibration targets so the simulated data looks like the real data: engage dates
  2016-10-20 to 2017-12-27, snapshot 2017-12-31, closed-deal win rate about 0.63, sales cycle
  1–138 days, about 5.7% Prospecting rows, and about 68% of open deals missing the account
  (closed deals always have one).
- The raw file format, so `src.data.crm.build(data_dir=...)` loads the output unchanged.

### Added at the project team's request
- 9 extra products, including a new GTS series (16 products in total).
- A target industry for each product (one of the 10 account sectors).
- A 5-year trend per industry (2011–2016).
- Competitor products, compared on recycled %, longevity and made-in-USA.
- Unit cost and annual R&D expense per product.

### Invented assumptions (no real-world basis)
- Every new attribute value: prices of new products, recycled %, longevity, unit cost, R&D,
  each industry's trend shape.
- Competitor attributes: our product's values plus random noise.
- The win-probability formula below: its coefficients and which interactions exist.

## 2. Generation steps

| Step | Script | Output |
|---|---|---|
| Products, competitors, cost, R&D | `scripts/simulation/make_simulated_products.py` | `products.csv`, `competitor_products.csv`, `product_rd.csv` |
| Industry trends | `scripts/simulation/make_industry_trends.py` | `industry_trends.csv` |
| Deals | `scripts/simulation/make_simulated_deals.py` | `sales_pipeline.csv`, `ground_truth.csv`, `deal_comparison.csv` |

For each deal, `make_simulated_deals.py`:

1. Draws an account using Dirichlet weights, so some accounts buy far more than others.
2. Draws a sales agent uniformly.
3. Draws an engage date skewed towards later dates (Beta distribution), so pipeline volume grows over time.
4. Draws a product. A product whose target industry matches the account's sector is 2.5× more likely.
5. Draws one of the product's two competitors and computes our-minus-rival differences in
   recycled %, longevity and made-in-USA.
6. Attaches the account sector's trailing trend features: the 5-year average growth, the slope
   and the latest growth. These use only years *before* the engage year, so there is no look-ahead.
7. Computes the win probability (Section 3) and draws Won or Lost.
8. Assigns timing:
   - The close date is 1–138 days after the engage date (Gamma distribution); won deals take 4 days longer.
   - Deals not closed by 2017-12-31, plus 12% "stalled" deals, stay Engaging with no close date.
   - About 5.7% of deals become Prospecting with no engage date.
9. Sets the close value: list price × U(0.85, 1.05) for Won deals, 0 for Lost deals.

## 3. Planted win-probability model

```
logit = product_effect + agent_effect                        (each drawn from N(0, 0.4))
      + 0.6  * sector_match
      + 0.05 * trend_5y_avg_growth
      + k * 0.5 * sector_match * (trend_5y_slope > 0)        interaction 1
      - k * 0.8 * (log(price) > 8 and revenue < median)      interaction 2: affordability
      + k * 0.7 * (series == GTS and region == West)         interaction 3
      + env(trend, k) * [ 0.02 * recycled_diff + 0.15 * longevity_diff
                          + 0.4 * usa_diff - 0.5 * log(price / competitor_price) ]
      + intercept

env(trend, k) = clip(1 + k * trend_5y_avg_growth / 10, 0.3 / k, 2.5 * k)
```

- The industry trend acts as the "environment": in a growing industry the head-to-head
  comparison with the rival matters more.
- The intercept is found by bisection so that the overall win rate is 0.63.
- `k` (`--interaction-strength`) scales the interactions:
  - k = 3 is the default "strong interaction" setting and reproduces `data/crm_simulated/`.
  - k = 1 gives the weaker original setting.

### Smooth-signal scenario (`--smooth-strength s`, default 0)

This scenario was **designed on purpose to favour neural models over trees**. It answers
"under which data conditions does TabNet beat Random Forest?", not "is TabNet better?".

It adds four continuous per-deal signals, drawn from a separate random stream and written to
`deal_signals.csv`. Each is assumed to be known at engage time:

- `discount_pct`: the quoted discount, U(0, 30).
- `quote_gap`: log of our discounted quote over the competitor's quote, with per-deal noise.
- `engagement_score`: N(0, 1).
- `days_since_contact`: Exp(mean 10).

```
logit += s * ( 2.0 * tanh(0.8 * (0.9*disc - 1.1*gap + 0.8*engage - 0.7*contact))
             + 0.8 * disc * engage )                  (signals standardised)
```

The effect is a smooth function of an oblique combination of the signals, plus a continuous
product term. Trees can only approximate it with many axis-aligned steps; neural nets fit it
directly. With s = 0 the signals are pure noise and all other files are unchanged.

## 4. Output files (`data/crm_simulated/`)

| File | Use |
|---|---|
| `sales_pipeline.csv`, `accounts.csv`, `products.csv`, `sales_teams.csv` | Model input in the raw format |
| `deal_comparison.csv` | Per-deal rival and differences; extra features |
| `deal_signals.csv` | Continuous per-deal signals (only in newer generator output); extra features |
| `ground_truth.csv` | True `win_prob`, latent outcome and trend features. **Answer key: do not use `win_prob` as a feature.** |
| `industry_trends.csv`, `competitor_products.csv`, `product_rd.csv` | Lookup tables |
| `model_results.csv`, `seed_results*.csv`, `seed_summary*.csv` | Model results (`_smooth` = smooth-signal scenario) |

The repository keeps one simulated dataset, `data/crm_simulated/`: 100,000 deals, interaction
strength k = 3, seed 582. Its `deal_comparison.csv`, `ground_truth.csv` and `model_results.csv` were
generated before `deal_signals.csv` existed, so the stored extended results do not use the signals.
The earlier 20,000-deal and 100,000-deal k = 1 sets were removed. To recreate one without
overwriting the default, pass `--interaction-strength 1 --out-dir <other dir>` (plus
`--n-deals 20000` for the small set).

## 5. Reproduce

```bash
python scripts/simulation/make_simulated_products.py
python scripts/simulation/make_industry_trends.py
python scripts/simulation/make_simulated_deals.py                     # defaults: 100,000 deals, k = 3, seed 582
.venv-crm/bin/python scripts/simulation/run_simulated_models.py
.venv-crm/bin/python scripts/simulation/run_simulated_seeds.py        # 5 seeds x k in {1, 3}, about 25 min on 4 CPUs
.venv-crm/bin/python scripts/simulation/run_simulated_seeds.py --strengths 1 --smooth 1 --tag _smooth   # about 13 min
python scripts/analysis/product_investment_report.py                # heuristic product ranking
```

The same seed always gives identical data.

## 6. Limitations

- The structure of the data (accounts, agents, products, dates) comes from the real CRM data,
  but **the win/loss mechanism is entirely invented**. Results show whether an algorithm can
  learn the planted structure, not how well it predicts real deals.
- Conclusions such as "non-linear models beat LR when interactions are strong" partly follow
  from how the data was designed.
- The smooth-signal scenario was chosen *because* it favours neural nets. Its result (TabNet
  beats RF) must be reported together with that design choice, never as a general finding.
- Each seed draws a new simulated world, while the model seed stays fixed at 42. Variation from
  TabNet and MLP initialisation is not measured.
- A data-driven generator such as CTGAN would rely on fewer hand-made assumptions. However, it
  has no ground truth, cannot create the new columns (competitors, trends, R&D), ignores date
  constraints and adds no information beyond the roughly 6,700 real closed deals. If tried, it
  should be evaluated train-on-synthetic, test-on-real (TSTR), with CTGAN fitted on the real
  training split only.
