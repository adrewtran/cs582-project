# CS 582 Group 2 — agent guidance

## Contract

Project: **Predicting CRM Sales Opportunities Using Machine Learning**. Original: `CS582_Group2_Original_CRM_Proposal.docx`; transcription: `CS582_Group2_Project_Proposal.md`. Read README and PLAN before editing. Leads/Bank/Telco experiments were removed on 2026-10-07 (recoverable from commit `c5bcfa6`); do not switch away from CRM to get higher scores.

Required: LR, RF, MLP, TabNet; Dummy prior control. Accuracy, Precision, Recall, F1, ROC-AUC and confusion matrices. Importance and optional SHAP are implemented. Autonomous agents are not a proposal requirement.

## Run

- Python 3.12; `python scripts/tools/setup_cpu.py` installs tested CPU dependencies.
- `.venv-crm/bin/python -m src.run_project` generates full results by chaining `src.train` (fit, select, calibrate, save models; no test scoring), `src.evaluate` (metrics/figures/explanations/checks from saved models) and `src.predict` (open-deal or `--input` CSV scores). Code does not generate documents; `CRM_IEEE_Paper.docx` and `CRM_Final.pptx` at the root are maintained by hand (removed generator on 2026-10-09 at the user's request).
- Code layout: `src/data` (load, split, preprocess), `src/models`, `src/evaluation`, `src/explain`; run-folder subpaths come only from `src/outputs.py`. Scripts live in `scripts/tools`, `scripts/simulation`, `scripts/analysis`. `src.train --data-dir` trains on other raw-format data (e.g. simulated); such runs go to `reports/crm/simulated/` (git-ignored) and are refused for `reports/crm/final/`.
- `.venv-crm/bin/python -m pytest -q` includes real CPU fits and full smoke integration.
- The Colab notebook was removed on 2026-10-09. `notebooks/{original,simulated}/{1_train,2_evaluate,3_predict}.ipynb` (added the same day) run the stages cell by cell by calling the step functions in `src/train.py`, `src/evaluate.py` and `src/predict.py`; the two folders differ only in `DATA_DIR`. Keep logic in `src/`, never copied into notebooks.
- `reports/crm/final/` is the only and authoritative results folder (subfolders data, models, metrics, explain, checks, predictions, figures). The user deleted the earlier Linux results on 2026-10-09 and asked to regenerate; current results are a macOS run, so RF/TabNet differ slightly from the Linux numbers.

## Data/evaluation

Raw data is read-only; use `src.data.crm.build()`. Exclude close_value, close_date, deal_stage and opportunity_id as predictors. Dates control label availability, stage labels outcome, IDs link evidence.

Use `src.data.split.asof_split()` for all final models: same-date rows together, training/validation outcomes known before next period. Fit preprocessors only on training. Select model/epochs/threshold/calibration with validation; test is exploratory evaluation. Do not refit a calibrated base model. Record closed-only bias, validation reuse, static snapshots and repeated accounts. Prior examination means this is not a pristine test.

Main open explanations are selected-calibrated-model reference sensitivities, not additive SHAP or causes. Raw RF TreeSHAP is a separate diagnostic with reconstruction checks. Priority is heuristic. Flag missing accounts; score only Engaging rows; never label open as Lost. Snapshot scores are not historical as-of predictions for each open record.

## Working agreements

- Do not commit, push, merge or submit unless asked. The user authorized verification, commit, push and PR on 2026-10-06. Target `thai-phan/cs582-project:main`; the existing authorized fork `adrewtran/cs582-project` supports the same cross-fork PR route used by PR #1. No merge or course submission is authorized.
- Preserve unrelated working changes; exclude them from CRM staging/PR.
- Non-CRM experiments and legacy CRM outputs (`reports/crm/temporal/`, root `open_deal_predictions.csv`, `src/baselines.py`, `archive/`) were removed at the user's request on 2026-10-07. New documents must consume actual final run outputs.
- Verify Google Slides import separately before claiming it was tested.
- Leave actual team contribution statements to member review; do not invent them.
- Load joblib bundles only from trusted sources.

Team: Hong Thai Phan, Nguyen Khanh An Tran, Hoang Thien Bao Bui.
