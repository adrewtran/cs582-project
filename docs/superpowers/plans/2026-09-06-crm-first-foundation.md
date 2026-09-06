# CRM-First Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository's primary, Colab-runnable workflow match the submitted CRM Sales Opportunities proposal.

**Architecture:** Keep the existing `Dataset` and scikit-learn pipeline abstractions, strengthen the CRM loader, and add CRM-specific orchestration rather than duplicating preprocessing in notebooks. The notebook calls tested source functions and exports reproducible reports.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, Matplotlib, pytest, Google Colab

**Spec:** `docs/superpowers/specs/2026-09-06-crm-first-design.md`

## Global Constraints

- CRM is the primary dataset.
- Use only engagement-time features for the honest model.
- Never include `deal_stage`, `close_date`, `close_value`, or `opportunity_id` in honest model features.
- Use `random_state=42`.
- Run on Google Colab Free without requiring a GPU.

---

### Task 1: Lock CRM data behavior

**Files:**
- Modify: `src/datasets/crm.py`
- Create: `tests/test_crm_dataset.py`

**Interfaces:**
- Produces: `build(data_dir: Path) -> Dataset` with chronological labeled rows and `extras["open_deals"]`.

- [ ] Write tests for product normalization, preserved join row count, leakage exclusion, closed target creation, chronological order, and Engaging-only scorable deals.
- [ ] Run `pytest tests/test_crm_dataset.py -v` and confirm the new expectations fail.
- [ ] Update the loader and report metadata minimally.
- [ ] Run the test file and confirm it passes.
- [ ] Commit the task locally.

### Task 2: Add honest chronological baselines

**Files:**
- Modify: `src/baselines.py`
- Create: `tests/test_crm_baselines.py`

**Interfaces:**
- Produces: `split(dataset, "temporal")` and `make_models(dataset)` including a Dummy classifier.

- [ ] Write tests that the temporal holdout is later than training data and all model names are present.
- [ ] Run the tests and confirm failure.
- [ ] Add DummyClassifier and date-aware temporal splitting without exposing the date as a feature.
- [ ] Run baseline tests and the full suite.
- [ ] Commit the task locally.

### Task 3: Add open-deal scoring

**Files:**
- Create: `src/crm_workflow.py`
- Create: `tests/test_crm_workflow.py`

**Interfaces:**
- Produces: `priority_group(probability: float) -> str` and `score_open_deals(model, dataset) -> DataFrame`.

- [ ] Write tests for probability bounds, priority thresholds, Engaging-only output, and retained opportunity IDs.
- [ ] Run the tests and confirm failure.
- [ ] Implement scoring with thresholds High >= 0.70, Medium >= 0.40, otherwise Low.
- [ ] Run the workflow tests and full suite.
- [ ] Commit the task locally.

### Task 4: Add the Colab entry point and user documentation

**Files:**
- Create: `notebooks/CRM_Sales_Opportunities.ipynb`
- Create: `README.md`
- Modify: `PLAN.md`
- Modify: `AGENTS.md`
- Modify: `requirements.txt`

**Interfaces:**
- Notebook consumes `src.datasets.load`, `src.baselines.run`, and `src.crm_workflow`.

- [ ] Create a top-to-bottom notebook with clone/install, data audit, baseline training, metric display, and open-deal export cells.
- [ ] Document exact Colab verification steps and expected output paths.
- [ ] Replace the pending dataset pivot with the owner-approved CRM-first decision.
- [ ] Add pytest to development requirements.
- [ ] Validate notebook JSON, run tests, and execute the source workflow.
- [ ] Commit the task locally.

### Task 5: Package the no-push handoff

**Files:**
- Create: `crm-first-update.patch`
- Create: `cs582-project-crm-first.zip`

**Interfaces:**
- Produces two uploadable artifacts because the connected GitHub account has read-only access.

- [ ] Confirm `git diff --check` is clean.
- [ ] Run the complete test suite.
- [ ] Run CRM data preparation and baselines.
- [ ] Generate the patch and ZIP without `.git` or virtual environments.
- [ ] Provide upload and Colab verification instructions.
