# Session log — 2026-09-02: from CRM to Lead Scoring, baselines on three datasets

Written by Claude Code at the end of the session. Facts only; see AGENTS.md for working rules and PLAN.md for
the dataset plan. Nothing was committed during this session — everything below is in the working tree.

## Timeline

1. **AGENTS.md created** for the repo (then only the proposal, guidelines, and Maven CRM CSVs).
2. **`src/data.py` written** — load/clean/join the four CRM tables. Profiling showed all 1,425 missing accounts and
   all 500 missing `engage_date`s were Engaging/Prospecting rows; the 6,711 Won/Lost rows join cleanly.
3. **Baselines (LR, RF) on CRM: chance level.** ROC-AUC 0.53 / 0.50. Verified it was the data, not the code:
   adding `close_value` gives AUC 1.000 (it is 0 for every Lost deal); win rate is ~63% in every category level;
   |corr| < 0.02 for every numeric.
4. **Tried `close_value` regression on Won deals (user's pick of option 1).** Also a dead end:
   `close_value = sales_price × iid noise (sd ≈ 0.10)`; predicting the list price gives R² 0.985 and nothing
   beats it. Days-to-close is equally flat. Every attribute in the Maven CRM data is independent of every outcome.
5. **Recommended swapping datasets.** User downloaded three: X Education Lead Scoring (Kaggle), UCI Bank
   Marketing, IBM Telco Churn. Quick RF/LR pass with and without each dataset's post-outcome columns:

   | Dataset | Rows | Positive | Leak columns | AUC with | AUC without |
   |---|---:|---:|---|---:|---:|
   | Leads | 9,240 | 38.5% | Tags, Lead Quality, Asymmetrique ×4, Last Notable Activity | 0.98 | 0.89 |
   | Bank | 41,188 | 11.3% | duration | 0.95 | 0.80 |
   | Telco | 7,043 | 26.5% | Churn Score, CLTV, Churn Reason | 0.97 | 0.85 |
   | CRM | 6,711 | 63.2% | close_value, close_date | 1.00 | 0.53 |

6. **PLAN.md written and published** (artifact: https://claude.ai/code/artifact/bacefe00-3b56-43fe-9e51-aee0ada29524).
   Decision: Leads primary, Bank secondary, Telco optional, CRM kept as a leakage-audit appendix. Novelty
   sharpened to: do TabNet's attention masks agree with SHAP on the tree model, across datasets?
7. **Housekeeping + Phase 1.** `data/` renamed to `leads/ bank/ telco/ crm/`; `src/` refactored to a generic
   `Dataset` + `src/datasets/<name>.py` loaders + `src.datasets.load(name)`; `regression.py` archived. Leads
   baselines: RF 0.901 / LR 0.888 ROC-AUC; without `Lead Profile` 0.889 / 0.875. Proposal's title, data-set
   paragraph, and references updated. Confirmed `torch` 2.14 and `pytorch-tabnet` 4.1 have cp314 wheels.
8. **Phase 2.** `bank.py`, `telco.py`, `--split temporal`, `src/summary.py`. Fixed a bug in `strip_strings`
   (`.str.strip()` NaN-ed the floats in Telco's mixed `Total Charges` column; now uses `map`).

## Final baseline table (`reports/summary.csv`)

| dataset | split | model | acc | bal.acc | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| leads | random | LR | 0.806 | 0.803 | 0.758 | 0.888 | 0.826 |
| leads | random | RF | 0.834 | 0.839 | 0.799 | 0.901 | 0.851 |
| bank | random | LR | 0.835 | 0.752 | 0.469 | 0.801 | 0.460 |
| bank | random | RF | 0.865 | 0.757 | 0.508 | 0.802 | 0.481 |
| bank | temporal | LR | 0.486 | 0.588 | 0.506 | 0.699 | 0.502 |
| bank | temporal | RF | 0.716 | 0.642 | 0.493 | 0.725 | 0.513 |
| telco | random | LR | 0.743 | 0.755 | 0.617 | 0.849 | 0.645 |
| telco | random | RF | 0.769 | 0.755 | 0.625 | 0.847 | 0.651 |

## Findings worth carrying into the paper

- **CRM leakage audit** (step 3–4): a clean negative result with a mechanism (`close_value` ≡ label).
- **Bank temporal drift**: first 80% of rows are 6.4% positive, last 20% are 30.8%. AUC 0.80 → 0.70–0.72, and the
  top features become `euribor3m`, `emp.var.rate`, `nr.employed` — era proxies. Random-split numbers overstate it.
- **`Lead Profile`** is borderline leakage (rep's assessment); worth ~0.012 AUC. Report both ways.
- Telco is flat between LR and RF (~0.85) — a fair test of whether MLP/TabNet add anything.

## Open decisions for the team

- Agree to the dataset swap (PLAN.md) and re-submit the updated proposal.
- Keep or drop `Lead Profile`.
- Whether Telco stays in scope.

## Next

Phase 3: `src/models/mlp.py` and `src/models/tabnet.py` using `src.baselines.split` and
`src.features.make_preprocessor`, run on all three datasets (Bank on both splits), rows appended to
`reports/<name>/baseline_metrics.csv` so `src.summary` picks them up. Then Phase 4 (SHAP, TabNet masks, rank
agreement).

## Environment

Python 3.14.6 in `.venv/`; pins in `requirements.txt` (pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0,
matplotlib 3.11.1, openpyxl 3.1.5). `torch` / `pytorch-tabnet` not yet installed.
