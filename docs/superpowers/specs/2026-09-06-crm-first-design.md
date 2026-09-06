# CRM-First Project Design

## Decision

CRM Sales Opportunities is the primary dataset because it is the dataset named in the submitted proposal. Leads, Bank, and Telco are outside the primary project and may remain only as archived experiments.

## Prediction task

Predict whether a closed opportunity is `Won` (1) or `Lost` (0) using only information available when the opportunity is engaging. Train on closed deals and score open `Engaging` deals as a demonstration. `Prospecting` rows are not scored because `engage_date` is not yet available.

## Leakage policy

Never use `deal_stage`, `close_date`, `close_value`, or `opportunity_id` as model features. The notebook will include an explicit leakage experiment showing why `close_value` creates a misleading near-perfect result, followed by the honest leakage-safe experiment.

## Models and evaluation

Use a chronological 80/20 holdout based on `engage_date`, with preprocessing fitted on training data only. Compare Dummy, Logistic Regression, Random Forest, MLP, and TabNet. Report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, runtime, and model limitations.

## Novel contribution

The project combines: (1) a reproducible leakage audit, (2) an honest traditional-versus-neural comparison on the same chronological split, and (3) explainable open-deal prioritization. The final output includes win probability, High/Medium/Low priority, and positive/negative factors where the chosen model supports local explanations.

## Runtime and files

Everything must run in Google Colab Free. Reusable code lives under `src/`; tests live under `tests/`; the main guided workflow lives in `notebooks/CRM_Sales_Opportunities.ipynb`; compact results live under `reports/crm/`.

## First implementation milestone

The first milestone makes the CRM path correct and verifiable: CRM-first documentation, validated data construction, chronological splitting, Dummy/LR/RF baselines, open Engaging-deal scoring, automated tests, and a Colab notebook. MLP, TabNet, SHAP, and final presentation figures follow after this foundation passes in a fresh Colab runtime.
