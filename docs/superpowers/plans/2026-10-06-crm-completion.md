# CRM Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the CRM proposal's code, experiment and presentation/report deliverables in one reproducible CPU run.
**Architecture:** Keep Dataset and train-fitted preprocessors; add strict dated evaluation, a shared model runner,
audits/explanations and result-driven document generation. Keep legacy results intact.
**Tech Stack:** Python 3.12, pandas, sklearn, CPU PyTorch/TabNet, SHAP, Matplotlib, python-pptx, python-docx, pytest.
**Spec:** `docs/superpowers/specs/2026-10-06-crm-completion-design.md`

## Global Constraints

- CRM only; raw data unchanged; seed 42; free CPU; no test-based model selection.
- No commit, push or merge in this run. Continue existing feature branch and preserve unrelated edits.
- User requested one-pass execution; do not stop for intermediate design/plan approvals.
- Every report states limitations; new results are exploratory and open scoring is a snapshot demo.

## Review Focus

- Late outcomes and equal-date boundaries must never leak labels into training/validation.
- Missing accounts/unseen categories and empty Engaging sets must produce explicit honest outputs.
- Calibration and explanations must refer to the same frozen model; no silent refit or SHAP mislabeling.
- CPU neural budgets/dependencies must fail clearly, not silently skip promised models.
- ZIP/Colab setup and generated papers/slides must use actual results rather than stale legacy files.

### Task 1: Data and as-of evaluation

**Files:** `src/datasets/crm.py`, `src/temporal.py`, `tests/test_temporal.py`, `tests/test_crm_validation.py`.
**Interfaces:** `asof_split(dataset) -> Split` with train/validation/test index arrays and manifest; `build()` remains compatible.
- [ ] Write failing tests for late labels, tied dates, invalid dates/stages/IDs and feature exclusions.
- [ ] Run tests; expected fail for missing split/validation.
- [ ] Implement deterministic date-group split and strict validation, retaining audit metadata.
- [ ] Run complete suite; expected pass.

### Task 2: Models, metrics and decision support

**Files:** `src/models.py`, `src/evaluation.py`, `src/explain.py`, `src/crm_workflow.py`, corresponding tests.
**Interfaces:** `fit_model(name, dataset, train_idx, val_idx, quick=False) -> FittedModel`; `metrics(y,p,threshold)`;
`choose_threshold(y,p)`; `CalibratedModel.fit(fitted, Xval,yval)`; `explain_open(model,dataset,reference,threshold)`.
- [ ] Write failing tests for five real model fits, normalization, threshold tie rules and metric values.
- [ ] Implement CPU models, train-only preprocessing, validation early stopping/selection and calibration.
- [ ] Test calibrated score consistency, reference explanation metadata, empty cases and LR additive arithmetic.
- [ ] Implement and run complete suite; expected pass, no skipped mandatory-model tests.

### Task 3: Full experiment and reproducible artifacts

**Files:** `src/analysis.py`, `src/run_project.py`, `tests/test_project_run.py`, requirements and notebook.
**Interfaces:** `run(output_dir,quick=False) -> dict`; artifacts include metrics, split/data/run JSON, predictions and figures.
- [ ] Write failing integration test executing all five models, leakage control and persisted outputs in temp directory.
- [ ] Add EDA, leakage experiment, held-out permutation/calibration/priority checks, SHAP and manifests.
- [ ] Implement CLI and fail-fast Colab notebook using shared entry point; validate JSON and execute notebook locally.
- [ ] Run full CPU experiment and full suite; expected all five models and 1,589 exported rows.

### Task 4: Documentation and deliverables

**Files:** `src/deliverables.py`, `README.md`, `PLAN.md`, `AGENTS.md`, `Group2_Project_Proposal.md`, reports/deliverables.
**Interfaces:** `build_deliverables(output_dir)` consumes run artifacts, creates paper Markdown/DOCX and PPTX + speaker notes.
- [ ] Write failing tests checking figures/table values and editable deck/report output from a real run.
- [ ] Generate evidence-based documents, synchronize CRM documentation and write detailed Colab/ZIP/Git instructions.
- [ ] Validate slides visually and document required team review/remaining external steps.
- [ ] Independent whole-change review; fix important findings with regression tests, re-run full suite and pipeline.
- [ ] Produce handoff ZIP/patch excluding environments, unrelated raw-file edits and old packages; no push.
