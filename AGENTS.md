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
data/<name>/              Raw datasets, committed, read-only: leads/ (primary), bank/, telco/, crm/ (retired)
data/<name>/processed/    Written by `python -m src.datasets <name>` — gitignored, regenerable
src/data.py               `Dataset` container + cleaning helpers shared by every loader
src/datasets/<name>.py    One loader per dataset; `src.datasets.load(name)` is the registry
src/features.py           ColumnTransformer built from a Dataset's column roles (one-hot, median-impute, opt. scale)
src/baselines.py          Logistic Regression + Random Forest: split (random|temporal), metrics, figures, importances
src/summary.py            Collects reports/*/baseline_metrics.csv into reports/summary.csv
archive/regression_crm.py CRM-only close_value regression (dead end; kept for the leakage-audit appendix)
reports/<name>/           Metrics table, importance CSVs, figures/ per dataset
PLAN.md                   Dataset decision, cleaning sheets, phases — read it before touching a loader
requirements.txt
Group2_Project_Proposal.md
Project Guidelines.md
```

Suggested placement for what does not exist yet: `src/models/{mlp,tabnet}.py`, `src/explain.py`, `notebooks/` for EDA.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.datasets leads     # clean + write data/leads/processed/, print the cleaning report
.venv/bin/python -m src.baselines leads    # fit LR + RF, write reports/leads/ (~1 min; bank ~3 min)
.venv/bin/python -m src.baselines bank --split temporal   # last 20% of rows as test → reports/bank/temporal/
.venv/bin/python -m src.summary            # every baseline_metrics.csv → reports/summary.csv
```

`src.datasets.load(name)` returns a `Dataset` (`.frame`, `.target`, `.categorical`, `.numeric`, `.features()`,
`.labels()`, `.report`). Build models on top of it, never on the raw CSVs — leak rules and join fixes live in the
loaders. The split (`src.baselines.split`: 80/20 stratified, `RANDOM_STATE = 42`) and preprocessor
(`src.features.make_preprocessor(dataset, scale_numeric)`) are shared; any new model must use both so its row in
`reports/<name>/baseline_metrics.csv` is comparable.

To add a dataset: write `src/datasets/<name>.py` with a `build() -> Dataset`, register it in `LOADERS`, and add a
cleaning sheet to PLAN.md. `Dataset.__post_init__` validates column roles and a 0/1 target.

## Data

Four datasets under `data/`; PLAN.md has the full comparison and cleaning sheets. Summary of the traps:

### Leads (primary) — `data/leads/Leads X Education.csv`, 9,240 × 37, target `Converted` (38.5%)

- `"Select"` is a form placeholder → NaN (`src/datasets/leads.PLACEHOLDERS`).
- **Leakage — never features:** `Tags`, `Lead Quality`, `Asymmetrique ×4`, `Last Notable Activity`. These are
  sales-team annotations written after contact (Tags "Ringing" converts 3%, "Will revert after reading the email"
  97%). Listed in `leads.LEAKAGE_COLUMNS`. With them, AUC ≈ 0.98; without, ≈ 0.89–0.90.
- **Borderline:** `Lead Profile` (Potential Lead 79%, Student of SomeSchool 4%) reads like a rep's assessment. Kept by
  default; `leads.build(include_lead_profile=False)` for the ablation. Report both in the paper.
- Missingness is informative → categoricals get a literal `Missing` level, not imputation. `How did you hear`
  (78% missing) and `Lead Profile` (74%) are mostly `Missing`.
- 5 constant columns and 6 near-constant Yes/No flags (< 10 minority rows) are dropped automatically — see the
  cleaning report.
- Column names are snake_cased in the `Dataset`; the raw file uses spaced titles.

### Bank (secondary) — `data/bank/bank-additional/bank-additional-full.csv`, 41,188 × 21, target `y` (11.3%)

- `sep=";"`. Use `bank-additional-full`, not `bank/bank-full.csv` (lacks the 5 macro columns).
- **Leakage:** `duration` — documented in `bank-additional-names.txt`. AUC 0.95 with, 0.80 without.
- `unknown` is a real category; `pdays = 999` = never contacted (`never_contacted` flag + NaN).
- **Rows are date-ordered (2008–2010) and the label drifts hard:** the first 80% of rows are 6.4% positive, the last
  20% are 30.8%. `python -m src.baselines bank --split temporal` drops ROC-AUC from 0.80 to 0.70–0.72 and the top
  features become the macro indicators (`euribor3m`, `emp.var.rate`, `nr.employed`), which are era proxies. Report
  both splits; the random-split number alone overstates what the model knows.

### Telco (optional) — `data/telco/Telco_customer_churn.xlsx`, 7,043 × 33, target `Churn Value` (26.5%)

- **Leakage:** `Churn Score`, `CLTV`, `Churn Reason`, `Churn Label`. `Total Charges` is a mixed object column (11
  blank strings among floats) — `src.data.strip_strings` uses `map`, not `.str.strip()`, precisely because the
  `.str` accessor turns the floats into NaN. Drop IDs and geography (all California).

### CRM (retired) — `data/crm/`, 6,711 labeled × 18 features, target `is_won` (63.2%)

- Every attribute is independent of every outcome: leakage-free AUC ≈ 0.50–0.53, and `close_value` on Won deals is
  `sales_price × iid noise`. `close_value` is 0 for every Lost deal → AUC 1.0 if leaked. Kept only for the paper's
  leakage-audit appendix. Join gotchas (`GTXPro` vs `GTX Pro`, `technolgy`, all missing accounts being open deals)
  are handled in `src/datasets/crm.py`.

### Cross-cutting

- Every dataset here has at least one column that is only known after the outcome. Any model reporting an AUC far
  above the leakage-free numbers in PLAN.md must be audited for leakage before being believed.
- `data/` is committed, read-only input. Never modify the raw files; loaders write to `data/<name>/processed/`.

## Conventions

- Python 3.14 with Pandas, NumPy, Matplotlib, Scikit-learn. `torch` 2.14 and `pytorch-tabnet` 4.1 have cp314
  wheels (verified 2026-09-02), so the advanced work stays on this venv.
- Dependencies are pinned in `requirements.txt`; use a local `.venv/` (already gitignored).
- Set and reuse a fixed `random_state` for every split and model so results in the paper are reproducible.
- Fit preprocessing inside a scikit-learn `Pipeline`/`ColumnTransformer` on the training fold only — no fitting on the
  full dataset before the split.
- Report the full metric set for every model (proposal's six + PR-AUC and balanced accuracy — Bank is 11%
  positive, so accuracy alone misleads). `class_weight="balanced"` is the default in the baselines.
- Keep the writing files (`*.md` at the repo root) as course deliverables; match their existing tone and formatting.

## Working agreements

- Team: Hong Thai Phan, Nguyen Khanh An Tran, Hoang Thien Bao Bui. `main` is the working branch.
- Do not commit or push unless asked.
- Prefer small, runnable scripts/notebooks that regenerate their outputs over one-off manual steps — the code is graded.
