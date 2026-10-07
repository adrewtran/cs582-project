# Verification record — 2026-10-06

Initial preparation used `codex/crm-first-foundation` at `52c3191ccfbbbbfc1f3312ebdf100091d449588a`, without commit/push. Authorized integration now uses `codex/crm-completion` based on upstream `main` at `f37826f90bebb7f7a33046222b81c28b09fa37d4`. Upstream PR #1 already incorporated the original CRM foundation; its historical report files are preserved unchanged in this PR.

## Executed evidence

- `python scripts/setup_cpu.py`: isolated Python 3.12 CPU environment installed; `pip check` reports no broken requirements.
- `.venv-crm/bin/python -m src.run_project`: full mode, all five models, SHAP, 1,589 scores, paper/slides/script. No mandatory model skipped.
- `.venv-crm/bin/python -m pytest -q`: 41 passed before final independent review; any later count is recorded below.
- `.venv-crm/bin/python scripts/verify_notebook.py`: all six unchanged code cells executed sequentially in one Python process, including the full pytest subprocess (43 passed) and full-budget experiment; output assertions passed. Exported executed notebook and JSON under `reports/crm/final/verification/`.
- Jupyter kernel startup was attempted but local TCP and IPC socket binding were prohibited. This is **not** a successful Jupyter or Colab kernel test. Login/upload/download UI remain unverified.
- PPTX and DOCX converted to PDF with LibreOffice. All 12 slides inspected in a contact sheet; the corrected output-example slide and paper pages inspected separately. Native Google Slides import remains unverified.
- Repeated full TabNet fits in the final CPU environment produced identical probabilities. A previous execution environment produced different early-stopping results despite identical seed/pins. README discloses this variability; no test-based model selection was added.

## Corrections verified during implementation

- Invalid data/date/join/ID inputs reject clearly; as-of split purges late outcomes and keeps same dates together.
- Legacy empty-open scoring returns an empty schema; LR contributions plus intercept reconstruct logits.
- All five real model fits return valid complementary probabilities; metric formulas and frozen calibration are tested.
- Serialization reproduces the exported open scores.
- A pandas Series attribute/method name collision had inserted method text into the example slide; a regression test now checks the actual ID/product from the exported CSV.
- Resolving the virtualenv interpreter symlink in the notebook escaped the environment. A regression test reproduced the wrong prefix; keeping the absolute unresolved path fixed it.
- CPU setup now supports an existing POSIX symlink-based environment as well as a fresh environment.

## Outside local verification

The group still must execute its Colab session, inspect Google Slides import, review actual member contributions and course submission requirements, and submit the work. Predictive utility is weak and is not certified by software tests. GitHub reports no direct push access on `thai-phan/cs582-project`, but write access on the existing `adrewtran/cs582-project` fork. This is the same PR route used by merged PR #1; the target repository remains unchanged.

## Independent review and final fixes

A fresh-context reviewer examined the full dirty working tree, not an empty commit range. No Critical leakage/calibration/packaging defect was found. Empty-open document failure and hardcoded generated counts were reproduced, then fixed with zero-row and changed-size output regressions. Notebook verification now follows run-derived counts. Missing notebook pytest execution and PR/neural-learning figures were completed as explicit design requirements.

Final suite after fixes: **43 passed**, with expected upstream TabNet deprecation/best-checkpoint warnings. The notebook now runs this suite before its full-budget experiment. New figures: precision–recall curves (AP labeled accurately), MLP and TabNet training-loss/validation-AUC histories.

Before fixes, the handoff patch applied to a separate clean clone at the base and produced exactly the packaged Git tree. ZIP CRC, raw CRM bytes, committed-original Telco preservation and unchanged real index checks passed. The final archive is regenerated and checked again after refreshing source hashes and documents.

Review boundaries accepted: no comprehensive security/license/hostile-input/concurrency audit, no regression audit of unrelated alternative datasets, and no verified Windows/macOS/cloud UI. Member authorship/contributions and final submission details require the team. No optional code-quality issues remain deferred; predictive/business limitations remain explicit.

## Pre-PR recheck — 2026-10-06

After updating to the latest upstream base and refreshing delivery wording, the suite passed again (43 tests), `pip check` passed, and all six notebook code cells completed another full-budget five-model run with 1,589 scored opportunities. The authoritative final output folder was regenerated separately and source/data hashes matched. The PR preserves upstream legacy CRM reports and all raw data; unrelated local Telco changes are excluded. Editable DOCX/PPTX are included; older rendered PDF previews and LibreOffice temporary files are excluded from this commit.

## Publication attempt

The verified implementation was committed locally as `ff8a35b26516acfa8884659c7e3ee0732e992066`. Terminal push to the existing project fork failed because no terminal GitHub credentials were available. The GitHub connector then rejected its first blob creation with HTTP 403, `Resource not accessible by integration`, despite repository metadata reporting account-level write access on that fork. That terminal/API attempt created no remote branch, commit or PR. Account permission and integration permission are distinct. The user subsequently approved a browser workflow, signed in, and the publication workflow continued through the GitHub UI. Final remote state must be checked separately. No credentials were requested or changed.
