# Dataset plan

Status: proposed 2026-09-02, pending team agreement. Housekeeping, Phase 1 and Phase 2 done the same day. Replaces the Maven CRM dataset, which turned out to have no
pre-close signal (see AGENTS.md → Data gotchas).

## Decision

| Role | Dataset | Why |
| --- | --- | --- |
| **Primary** | Lead Scoring (X Education) — `data/leads/` | Same shape as the proposal: a lead/opportunity → converted or not. 9,240 rows, 38.5% positive, leakage-free AUC ≈ 0.89 so there is headroom for MLP/TabNet to matter. Real messiness (`Select` placeholders, 15–50% missing) gives an honest cleaning story. |
| **Secondary** | Bank Marketing (UCI, `bank-additional-full.csv`) — `data/bank/` | 41,188 rows — the scale TabNet needs to show anything over trees. 11.3% positive forces imbalance handling. A documented leak (`duration`) and a documented benchmark literature to compare against. |
| **Optional** | Telco Customer Churn (IBM) — `data/telco/` | 7,043 rows, 26.5% positive, leakage-free AUC ≈ 0.85. Framing is churn, not sales; only worth adding as a third point in the cross-dataset explainability comparison if Phases 1–4 finish early. |
| **Retired** | Maven CRM — `data/crm/` | Keep as an appendix: a worked example of a leakage audit (close_value = label, AUC 1.00 → 0.50). |

Quick-look numbers (RF/LR, 80/20 stratified, `random_state=42`, no tuning):

| Dataset | Rows | Positive | Leak columns | AUC with leak | AUC without |
| --- | ---: | ---: | --- | ---: | ---: |
| Leads | 9,240 | 38.5% | Tags, Lead Quality, Asymmetrique ×4, Last Notable Activity | 0.98 | **0.89** |
| Bank | 41,188 | 11.3% | duration | 0.95 | **0.80** |
| Telco | 7,043 | 26.5% | Churn Score, CLTV, Churn Reason | 0.97 | **0.85** |
| CRM | 6,711 | 63.2% | close_value, close_date | 1.00 | 0.53 |

## Framing (proposal changes)

Only the *Data set* paragraph of the proposal changes. Novelty gets sharper, not different: compare LR / RF / MLP /
TabNet on two-to-three tabular conversion datasets, and test whether **TabNet's built-in attention masks agree with
SHAP on the tree model** — i.e. does the "interpretable" deep model tell the sales team the same story. The CRM
negative result and the per-dataset leak columns become a short "leakage audit" section, which is a genuine
methodological contribution for a sales-prediction paper.

## Cleaning sheets

### Leads (`Leads X Education.csv`, 9,240 × 37, target `Converted`)

- `"Select"` is a form placeholder, not a value → NaN (Specialization 1,942; How did you hear 5,043; Lead Profile
  4,146; City 2,249).
- Drop identifiers: `Prospect ID`, `Lead Number`.
- Drop 5 constant columns: Magazine, Receive More Updates…, Update me on Supply Chain…, Get updates on DM…, I agree to
  pay… by cheque.
- **Drop leakage** — sales-team annotations written *after* contact: `Tags` (0.97 vs 0.03 conversion by value),
  `Lead Quality`, `Asymmetrique Activity/Profile Index/Score`, `Last Notable Activity`.
- **Borderline: `Lead Profile`** (Potential Lead 0.79, Student of SomeSchool 0.04). It reads like a rep's assessment.
  Decide in EDA; report results with and without it.
- Missingness is informative here (a lead that skipped the form is different from one that filled it) → encode NaN as
  its own category for categoricals rather than imputing. Numerics (TotalVisits, Page Views: 1.5% missing) → median.
- Collapse `Country`, `City`, `Lead Source` long tails to top-k + Other before one-hot.
- Yes/No columns → 0/1.

### Bank (`bank-additional-full.csv`, 41,188 × 21, target `y`)

- Read with `sep=";"`. Use `bank-additional-full`, not the older `bank-full` (it lacks the 5 macro-economic columns).
- **Drop `duration`** — the names file says so explicitly: it is unknown before the call and near-determines `y`.
- `unknown` is a legitimate category (default 8,597, education 1,731) — keep it.
- `pdays = 999` means never contacted → add a binary flag and set the value to NaN/-1.
- 11.3% positive → `class_weight="balanced"` for LR/RF, and report **PR-AUC and balanced accuracy** alongside the
  proposal's metrics for this dataset.
- Rows are ordered by date (May 2008 → Nov 2010) and the macro columns drift → besides the random split, run one
  **temporal split** (last 20% by row order) as a robustness check.

### Telco (`Telco_customer_churn.xlsx`, 7,043 × 33, target `Churn Value`)

- `Total Charges` has 11 blank strings → numeric with NaN.
- Drop identifiers and geography: CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude
  (all rows are California; geo is noise at this size).
- **Drop leakage**: `Churn Score` (IBM's own model output), `CLTV`, `Churn Reason` (only exists for churners),
  `Churn Label` (duplicate of the target).

## Code changes

```
src/
  data.py            keep Dataset dataclass; categorical/numeric lists move onto the instance
  datasets/
    __init__.py      load(name) registry: "leads" | "bank" | "telco" | "crm"
    crm.py           today's src/data.py logic, DATA_DIR → data/crm
    leads.py bank.py telco.py
  features.py        make_preprocessor(dataset, scale_numeric) reads the lists from the Dataset
  baselines.py       --dataset flag; writes reports/<dataset>/
  models/mlp.py, models/tabnet.py
  explain.py         importances, SHAP, TabNet masks, cross-model rank agreement
```

Housekeeping (done 2026-09-02): rename `data/lead scoring/` → `data/leads/`, `data/bank+marketing/` → `data/bank/` (drop the
`.Rhistory`), move the Telco xlsx to `data/telco/`; CRM loader moved to `src/datasets/crm.py`; `src/regression.py` archived as
`archive/regression_crm.py`.

## Phases

1. **Now → week-4 milestone — done 2026-09-02.** Housekeeping, `datasets/leads.py`, baselines on Leads
   (`reports/leads/`: RF ROC-AUC 0.901 / LR 0.888; without `Lead Profile` 0.889 / 0.875), proposal paragraph updated.
2. **Bank + Telco loaders — done 2026-09-02.** Baselines on all datasets (`reports/summary.csv`). Finding: Bank's
   random-split AUC 0.80 falls to 0.70–0.72 on a temporal split because the label rate jumps from 6.4% (2008–09) to
   30.8% (2010) and the macro columns act as era proxies. Both splits go in the paper.
3. **MLP + TabNet.** Same split and preprocessor. (`torch`/`pytorch-tabnet` cp314 wheels confirmed.)
4. **Explainability.** RF importances, LR coefficients, TabNet masks, SHAP on the best model per dataset; Spearman
   agreement of top-10 feature rankings across models.
5. **Weeks 13–15.** Slides/video. 6. **Week 16.** Paper (CRM audit as appendix) and code cleanup.

## Risks

- `Lead Profile` may be a leak; results must be reported both ways until settled.
- TabNet is known to need large data — it may lose to RF on Leads and Telco. That is a result, not a failure; Bank is
  there to give it a fair shot.
- ~~PyTorch on Python 3.14~~ — verified: `torch` 2.14 and `pytorch-tabnet` 4.1 publish cp314 wheels.
