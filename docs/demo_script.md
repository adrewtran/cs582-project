# CRM Sales Opportunities: One-Page Demo Script (about 10 min)

Team: Hong Thai Phan, Nguyen Khanh An Tran, Hoang Thien Bao Bui

## Before you start
- Python 3.12 environment is set up (`python scripts/setup_cpu.py`).
- Open `reports/crm/final/` in a file browser. Open `figures/` and `deliverables/` in tabs.
- Terminal is in the repo root.

## 1. Problem (1 min)
**Say:** "We predict whether a CRM sales opportunity will be Won or Lost, and we show which factors influence the score. The input is four CRM tables: pipeline, accounts, products and teams."
**Show:** `README.md`, section 1.

## 2. Pipeline run (1 min)
**Run:** `.venv-crm/bin/python -m src.run_project --quick`
**Say:** "This is a smoke run with small budgets. It shows the pipeline works end to end, and its numbers are not reported. Our reported results come from the full run already saved in `reports/crm/final/`."
**Show:** `run_manifest.json` with `status: complete`, `mode: full`.

## 3. Data and leakage control (1.5 min)
**Show:** `leakage_audit.csv`, `data_quality.json`.
**Say:** "We excluded close_value, close_date, deal_stage and opportunity_id from the predictors, because they reveal the outcome or only identify the deal. We trained only on closed deals, and never treated open deals as Lost."

## 4. Time-based split (1 min)
**Show:** `split_manifest.csv`.
**Say:** "We split by time: train 2,975, validation 583, test 1,361. We purged 1,792 rows whose outcomes weren't known at the cutoff. Preprocessing was fit on the training set only."

## 5. Models and results (2 min)
**Show:** `validation_metrics.csv`, `test_metrics.csv`, `figures/` (ROC and confusion matrices).
**Say:** "We trained Dummy, Logistic Regression, Random Forest, MLP and TabNet. Logistic Regression was chosen on validation ROC-AUC, before we looked at test. Test ROC-AUC is about 0.50 to 0.52 for all models. The Dummy model has high accuracy and F1 only because it always predicts Won."
**Show:** `calibration_test.csv`.
**Say:** "Calibration improved Brier from 0.2794 to 0.2443, which is still worse than Dummy at 0.2421."

## 6. Explanations (1.5 min)
**Show:** `feature_importance_logistic_regression.csv`, `permutation_importance_test.csv`, `rf_shap_global.csv`, `shap_audit.json`.
**Say:** "These are sensitivities of the model's score, not causes. The Random Forest SHAP values are a separate diagnostic, and the additivity check passes with error under 1e-5."

## 7. Scoring open deals (1 min)
**Show:** `open_deal_predictions.csv`.
**Say:** "We scored 1,589 Engaging deals. Win and loss probabilities sum to 1, and 1,088 rows with a missing account are flagged. Priority is a heuristic, and these are snapshot scores, not historical predictions."

## 8. Deliverables (0.5 min)
**Show:** `deliverables/` (PPTX, DOCX, Markdown paper, ESL script).

## 9. Honest close (0.5 min)
**Say:** "Our results do not yet show the model helps sales prioritisation. Limitations: closed-only bias, validation reuse, static snapshots, repeated accounts, and the data was examined in earlier attempts, so the test set isn't pristine. The value of the project is a rigorous, leakage-free pipeline."

## Likely questions
- **Why are the scores so low?** The available features carry little signal. We did not add close_value to inflate the numbers.
- **Why did the Dummy model get high accuracy?** About 60% of closed deals are Won, so always predicting Won scores well.
- **Did you test on Colab?** Only if the team has run it. Run the notebook once beforehand and say what you verified.
- **Will TabNet give exactly the same numbers?** Not guaranteed across platforms. Seeds don't ensure identical results.

## Backup
If the live run is slow or fails, skip step 2 and demo only from `reports/crm/final/`.
