# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

CS 582 (Machine Learning) group project, Group 2: **Predicting CRM Sales Opportunities Using Machine Learning**.

Goal: a binary classification pipeline that predicts whether a CRM sales opportunity is **Won** or **Lost**, plus
explainability of *why* a given opportunity scores high or low.

- Baselines: Logistic Regression, Random Forest.
- Advanced: MLP, TabNet.
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC.
- Explainability: model feature importance, and SHAP if time permits.

See `Group2_Project_Proposal.md` for the approved scope and `Project Guidelines.md` for grading/deliverables. Treat the
proposal as the contract — do not silently change the modeling plan; raise it instead.

Deliverables: proposal, slides/video (week 15), paper (week 16), code and scripts (week 16).

## Layout

```
data/                     Raw CRM dataset (Maven Analytics), committed as CSV — read-only inputs
data/processed/           Derived tables written by `python -m src.data` — gitignored, regenerable
src/data.py               Load / clean / join / label the four raw tables
requirements.txt
Group2_Project_Proposal.md
Project Guidelines.md
```

Suggested placement for what does not exist yet:

- `src/` for the rest of the reusable modules (features, training, evaluation).
- `notebooks/` for EDA and reporting narratives.
- `reports/figures/` for generated charts referenced by the paper/slides.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.data          # writes data/processed/, prints a cleaning report
```

`src/data.build_dataset()` returns a `Dataset` with `.labeled`, `.open_deals`, `.features()`, `.target()`, and a
`.report` dict of counts. Build models on top of it rather than re-reading the raw CSVs — the join fixes and the
leakage rules live in one place, and `FEATURE_COLUMNS` / `LEAKAGE_COLUMNS` are the source of truth for what a model may
see.

## Data

Four raw tables in `data/`, described by `data/data_dictionary.csv`:

| File | Rows | Key |
| --- | --- | --- |
| `sales_pipeline.csv` | 8,800 | `opportunity_id` (fact table) |
| `accounts.csv` | 85 | `account` |
| `products.csv` | 7 | `product` |
| `sales_teams.csv` | 35 | `sales_agent` |

Joins: `sales_pipeline.account → accounts.account`, `sales_pipeline.product → products.product`,
`sales_pipeline.sales_agent → sales_teams.sales_agent`.

### Known data gotchas — handle these explicitly, do not paper over them

- **Product key mismatch.** `sales_pipeline` uses `GTXPro`; `products` uses `GTX Pro`. Normalize before joining or ~1/7
  of product joins silently drop. `src/data.PRODUCT_NAME_FIXES` handles it.
- **Label filtering.** `deal_stage` has four values: Won (4,238), Lost (2,473), Engaging (1,589), Prospecting (500).
  Only Won/Lost are labeled outcomes; the 2,089 still-open deals must be excluded from train/test. That leaves 6,711
  labeled rows at ~63% Won, so report Precision/Recall/F1 and ROC-AUC rather than leaning on accuracy.
- **Missing values sit entirely in the open deals.** 1,425 rows have an empty `account` and 500 have no `engage_date`
  — all of them Engaging/Prospecting. The 6,711 labeled rows join cleanly and have no missing features, so imputation
  is not needed for the Won/Lost modeling set. Re-check this if the label definition ever widens.
- **Leakage.** `close_value` and `close_date` are only known *after* the deal closes. Never use them as features.
  `engage_date` is safe; derived durations that depend on `close_date` are not.
- **Dirty categoricals.** `accounts.sector` contains typos (e.g. `technolgy`). Clean, don't drop — see
  `src/data.SECTOR_FIXES`.
- **Dates** are `M/D/YY` strings; parse with an explicit format.

`data/` is committed input. Do not modify or overwrite the raw CSVs — write derived data to a separate directory.

## Conventions

- Python 3 with Pandas, NumPy, Matplotlib, Scikit-learn (+ `pytorch-tabnet`, `shap` for the advanced work).
- Dependencies are pinned in `requirements.txt`; use a local `.venv/` (already gitignored).
- Set and reuse a fixed `random_state` for every split and model so results in the paper are reproducible.
- Fit preprocessing inside a scikit-learn `Pipeline`/`ColumnTransformer` on the training fold only — no fitting on the
  full dataset before the split.
- Report the full metric set listed above for every model, so the comparison table stays like-for-like.
- Keep the writing files (`*.md` at the repo root) as course deliverables; match their existing tone and formatting.

## Working agreements

- Team: Hong Thai Phan, Nguyen Khanh An Tran, Hoang Thien Bao Bui. `main` is the working branch.
- Do not commit or push unless asked.
- Prefer small, runnable scripts/notebooks that regenerate their outputs over one-off manual steps — the code is graded.
