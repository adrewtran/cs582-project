# CS 582 — Predicting CRM Sales Opportunities

Group 2: **Hong Thai Phan · Nguyen Khanh An Tran · Hoang Thien Bao Bui**.

The project keeps the CRM topic reported to the professor. The input is four CRM tables; the two main outputs are the **Won/Lost probability** and **the factors that influence the predicted score**. Priority is an illustrative label, not a validated sales policy.

**Getting started:** use `notebooks/CRM_Sales_Opportunities.ipynb` and the ZIP containing the latest source. You do not need GitHub push access, a GPU or a powerful local machine to run it.

## 1. Actual results

The full version was run in a Python 3.12 / CPU environment (macOS arm64; versions and hashes in `run_manifest.json`). Dummy, Logistic Regression (LR), Random Forest (RF), MLP and TabNet were all actually trained. The main results are in **`reports/crm/final/`**.

| Model | Test ROC-AUC | Accuracy | F1 (Won) |
|---|---:|---:|---:|
| Dummy prior | 0.5000 | 0.6003 | 0.7502 |
| Logistic Regression | 0.5238 | 0.4372 | 0.2620 |
| Random Forest | 0.5180 | 0.5922 | 0.7422 |
| MLP | 0.4954 | 0.4849 | 0.5135 |
| TabNet | 0.5068 | 0.5356 | 0.6368 |

LR was selected by **validation ROC-AUC**, before looking at test. The threshold maximises macro-F1 on validation; it is not the default 0.5 and is not tuned for F1(Won) on test. The Dummy's high Accuracy/F1 comes from always predicting Won. The results **do not yet demonstrate useful sales prioritisation**; `close_value` was not added to the inputs to artificially improve the numbers.

Calibrating LR changes the test Brier score from 0.2794 → 0.2443, still worse than the Dummy's 0.2421 (lower is better). Calibration does not increase ROC-AUC. This is a study with limitations, not a production model.

## 2. Run on Google Colab — step by step

Colab has a free tier, but quota/resources are not guaranteed: [Google FAQ](https://research.google.com/colaboratory/faq.html). The project only needs a CPU; there is no need to buy Pro.

1. Download the handoff source ZIP and `CRM_Sales_Opportunities.ipynb`. Keep the ZIP for uploading; you do not need to install Python on your laptop.
2. Open [Google Colab](https://colab.research.google.com/) and sign in with your Google account.
3. Choose **File → Upload notebook** and select the `.ipynb` you downloaded.
4. **Runtime → Change runtime type**: hardware accelerator **None/CPU**. The tested environment is **Python 3.12**. If Google changes the version, choose a 3.12 runtime if available; setup stops with a clear message on untested versions.
5. Choose **Runtime → Run all**.
6. The first cell asks for an upload: choose **exactly one ZIP of the latest source code**, not a `.patch` or a ZIP that only contains results.
7. Wait for setup and the pipeline to finish. The first dependency download can take a few minutes. The notebook uses its own Python environment; there is no need to restart the kernel between cells.
8. The last cell downloads a results ZIP. Save it before Colab ends the session; runtime storage is temporary.

The notebook calls the same command as the CLI. The code cells were executed sequentially with Python in the development environment. That environment blocks Jupyter kernel sockets, so **a Jupyter kernel or the team's signed-in Colab session has not been verified**. The team still needs to sign in, choose the upload/download files and confirm it once.

### Once the latest source has been pushed

You can choose the **GitHub** tab in Colab, search for `thai-phan/cs582-project` and pick **the correct feature branch or latest commit**. Before the PR is merged, `main` may still contain old code. Opening the notebook from GitHub does not load the whole repo into the runtime; the notebook still supports uploading a ZIP.

To clone instead of uploading: add a cell before the first cell and replace the placeholder:

```python
import os, subprocess
from pathlib import Path
PROJECT_DIR = Path('/content/cs582-project')
if PROJECT_DIR.exists():
    raise RuntimeError('Use a fresh runtime to avoid mixing two versions of the code.')
subprocess.run(['git', 'clone', 'https://github.com/thai-phan/cs582-project.git', str(PROJECT_DIR)], check=True)
subprocess.run(['git', '-C', str(PROJECT_DIR), 'checkout', 'PUSHED_BRANCH_OR_COMMIT'], check=True)
os.environ['CRM_PROJECT_ROOT'] = str(PROJECT_DIR)
```

There is no need to change ChatGPT's `adrewtran` GitHub connection. A private repo may need separate authentication to clone; uploading the ZIP avoids that step.

## 3. Run from the terminal (optional)

In the folder containing this README, using Python 3.12:

```bash
python scripts/tools/setup_cpu.py
.venv-crm/bin/python -m src.run_project
.venv-crm/bin/python -m pytest -q
```

After setup, **the single full-run command** is `.venv-crm/bin/python -m src.run_project`. It chains four stages, and each one also runs on its own:

| Stage | Command | Reads | Writes |
|---|---|---|---|
| Train | `python -m src.train` | `data/crm/` | `data/`, `models/`, `metrics/validation_metrics.csv` |
| Evaluate | `python -m src.evaluate` | saved models | `metrics/`, `explain/`, `checks/`, `figures/` |
| Predict | `python -m src.predict` | `models/model_bundle.joblib` | `predictions/open_deal_predictions.csv` |
| Documents | `python -m src.reporting.deliverables reports/crm/final` | all of the above | `deliverables/` |

Train fits all five models, selects one on validation, calibrates it and saves everything; no test row is scored there. Evaluate and predict only load the saved models, so they can be rerun without retraining. A failing step makes the command return an error and the manifest records `failed`; no model is silently dropped.

On Windows, the interpreter path is `.venv-crm\Scripts\python.exe`. Confirmed environments: Linux CPU (earlier run) and macOS arm64 CPU (current results); Windows is not claimed as tested.

```bash
# Smoke test: tiny budgets, do NOT report these results
.venv-crm/bin/python -m src.run_project --quick
# Keep a separate run
.venv-crm/bin/python -m src.run_project --output reports/crm/my_run
# Experiment only
.venv-crm/bin/python -m src.run_project --no-documents
# Recompute metrics, figures and checks from the saved models
.venv-crm/bin/python -m src.evaluate
# Score new deals given in sales_pipeline.csv format (needs engage_date)
.venv-crm/bin/python -m src.predict --input new_deals.csv --save new_deal_scores.csv
# Regenerate the paper/slides from existing results
.venv-crm/bin/python -m src.reporting.deliverables reports/crm/final
```

`--quick` writes `reports/crm/smoke/` with the label **SMOKE_TEST_NOT_FINAL**. Rerunning into the same output folder overwrites artifacts with the same name; use a different `--output` to keep the previous run. Files you added yourself are not deleted.

## 4. Verify — what does a correct run look like?

| Check | Result with the bundled data |
|---|---|
| `run_manifest.json` | `status: complete`, `mode: full`, stages train/evaluate/predict |
| `metrics/test_metrics.csv` | 5 models, all metrics, no NaN |
| Raw data | 8,800 pipeline; 85 accounts; 7 products; 35 teams |
| Labels | 6,711 closed: 4,238 Won / 2,473 Lost |
| Split | train 2,975; validation 583; test 1,361; purged 1,792 |
| `predictions/open_deal_predictions.csv` | 1,589 rows; win_probability + loss_probability = 1 |
| Missing account | 1,088 rows flagged |
| `explain/shap_audit.json` | RF additivity error < 1e-5 |
| `deliverables/` | PPTX, DOCX, Markdown and ESL script |

The notebook checks the main conditions itself. The tests also check date availability, tied dates, invalid IDs/stages/dates/joins, metric formulas, all five real models, calibration and that a reloaded bundle gives the same predictions. **Passing tests does not mean the model is good enough for business use.** The same code gives different RF/TabNet numbers on different platforms: the earlier Linux run had RF 0.5194 and TabNet 0.5173 test AUC, this macOS run 0.5180 and 0.5068; LR and MLP match exactly. Seeds do not guarantee bitwise-identical results across all platforms; early stopping may choose a different epoch. Do not pick the better result to decide the model: all models are weak, and LR is still the one selected on validation. Compare versions, hashes and platform when reproducing.

## 5. Data and feature policy

Source: [Maven CRM Sales Opportunities](https://mavenanalytics.io/data-playground/crm-sales-opportunities), a fictitious B2B company selling computer hardware. The four CSVs and the dictionary are in `data/crm/`; raw files are kept unchanged. Row counts are computed from the files, not taken from the website. The manifest stores SHA-256 hashes.

- Join on product, account and sales_agent, checking many-to-one relationships and that the row count does not grow.
- The loader normalises `GTXPro → GTX Pro` and `technolgy → technology`.
- Categorical: product, series, sector, office_location, regional_office, manager, sales_agent.
- Numeric: sales_price, revenue, employees, revenue_per_employee, year_established, account_age_at_engage, engage_year/month/quarter/dayofweek, is_subsidiary.
- **Excluded from predictors:** close_value, close_date, deal_stage, opportunity_id. close_date only checks label availability; stage creates the target; the ID is for traceability.
- Train only on Won/Lost; open deals are never labelled Lost. Score Engaging deals; exclude the 500 Prospecting deals that have no engage_date.
- Imputation, encoding and scaling are fit on training only; the other sets are only transformed.

The new split keeps same-date deals in the same period and purges outcomes not yet known at the cutoff. The dataset was examined in earlier attempts, so it is called an **exploratory holdout**, not a completely fresh test. The account/team/product tables are snapshots, with no guarantee that each value was correct at the historical time. Accounts/agents may repeat across periods; completely new customers have not been evaluated separately.

## 6. Outputs and novelty

| Column/file | Meaning |
|---|---|
| win_probability / loss_probability | Calibrated P(Won) / 1 − P(Won) |
| predicted_outcome | Threshold from validation, mapped through calibration |
| priority | High ≥0.70; Medium ≥0.40; Low <0.40; unvalidated heuristic |
| positive_factors / negative_factors | Inputs that make the probability higher/lower compared with the training reference |
| reference_deltas_json | All differences; **not SHAP, and they do not add up to the prediction** |
| input_warning / account_missing | Missing-account warning; imputation/unknown categories are not hidden |
| scoring_context | `frozen_model_snapshot_demo`, not a historical replay |
| rf_shap_* | SHAP for the **raw RF** only, not to be mislabelled as an explanation of the calibrated LR |
| models/model_bundle.joblib | Frozen model + calibration + features/reference/threshold (used by `src.predict`) |
| models/trained_models.joblib | All five fitted models + thresholds (used by `src.evaluate`) |

Reference sensitivity replaces each input with the training median/mode and measures the change in P(Won). This describes model behaviour; it does not prove that changing the input would change the real outcome. Replacing only one correlated feature can create unrealistic combinations. Only load joblib files from a trusted project, because this format deserialises Python objects.

Comparison of the promised novelty and why the results are still weak: [`docs/NOVELTY_AND_RESULTS.md`](docs/NOVELTY_AND_RESULTS.md).

**Novelty for the course project:** an audited, explainable evaluation workflow in which each safeguard has its own evidence:

| Contribution | Evidence |
|---|---|
| Prediction + explanation | Explanation checks (`explanation_checks.json`): on the raw RF, reference sensitivity agrees with TreeSHAP (median per-row Spearman 0.76; top-3 overlap 61% vs 17% by chance). Resetting a deal's top-3 inputs moves the selected model's P(Won) 3.1× more than resetting 3 random inputs. |
| Leakage control | `close_value` alone gives AUC 1.0; the honest model gives 0.524 (`leakage_audit.csv`). |
| Label-availability split | Compared with a no-purge split and a random split (`split_protocol_summary.csv`). The random split is at most +0.036 AUC higher, and every paired bootstrap interval for the no-purge change includes 0. All protocols stay near chance, so the split is a safeguard with a small measured effect here. |
| Input-quality warnings | 1,088 of 1,589 open-deal scores are flagged for a missing account. |
| Known-truth synthetic benchmark | Same pipeline on simulated deals with a planted win formula reaches AUC 0.875 vs an oracle of 0.886 (`data/crm_simulated/`, `docs/SIMULATED_DATA.md`). This shows the pipeline learns when signal exists. Simulated scores are never real-world performance. |

No new algorithm, causal effect or proven revenue increase is claimed. The 3-month expected revenue on the final slides is illustrative only and has not been validated.

## 7. Files for reporting

All in `reports/crm/final/`:

- `deliverables/CRM_Final_Report.docx` and `.md`: method, results, limitations, related work and references.
- `deliverables/CRM_Final_Presentation.pptx`: 12 editable slides, Arial, speaker notes. Upload to Drive → Open with Google Slides; check the layout after import. [Google guide](https://support.google.com/docs/answer/9310378?hl=en).
- `deliverables/SPEAKER_SCRIPT_ESL.docx` and `.md`: 3 people × 4 slides, about 8–10 minutes; includes Q&A.
- `deliverables/TEAM_REVIEW.md`: steps the team needs to confirm before submission.
- `metrics/`: validation and test metrics, calibration, raw test probabilities.
- `checks/`: leakage control, priority bands, split-protocol comparison and explanation checks (evidence for the novelty claims).
- `explain/`: native and permutation importance, RF TreeSHAP and its additivity audit.
- `predictions/`: open-deal scores with factors and warnings; `expected_revenue_3m.json` from `scripts/analysis/expected_revenue_3m.py`.
- `figures/`: ROC, confusion matrices, EDA, calibration, importance, SHAP.
- `data/split_manifest.csv`, `models/model_training.json`, `run_manifest.json`: row assignments, configurations, versions and hashes.

The 1–8 slide limit in the earlier announcement applies to the progress forum. This 12-slide deck is for the final; check the final limit if the professor announces one separately. No video has been recorded, nothing has been submitted, and members' actual contributions have not been written in.

## 8. Commitments → evidence

| Commitment | Delivered |
|---|---|
| CRM Won/Lost + join/clean/EDA | `src/data/crm.py`, `src/evaluation/figures.py`, data-quality/EDA outputs |
| LR, RF, MLP, TabNet | `src/models/zoo.py`, 5 metric rows including the control |
| Accuracy/Precision/Recall/F1/AUC/confusion | `src/evaluation/metrics.py`, test metrics and figures |
| Feature importance + optional SHAP | native/permutation CSV, RF SHAP and additivity audit |
| Explaining inputs/decisions | per-deal explanation, missing-input warning |
| Reproducible/free CPU | setup, runner, notebook, tests |
| Paper + slides | editable team-review drafts |

`docs/COMMITMENT_AUDIT.md` stores the initial audit and the remediation status. Original CRM proposal: `docs/CS582_Group2_Original_CRM_Proposal.docx`; the Markdown file at the root is a transcription. The old Leads/Bank/Telco directions were removed from the repo (recover them from commit `c5bcfa6` if needed).

## 9. Push the branch / open a PR to the team repo

The target repo is `thai-phan/cs582-project`, base `main`. The PR route used for this project is from the existing fork `adrewtran/cs582-project`; this fork has push access, although the current connection cannot yet push directly to the target repo. Handoff branch: `codex/crm-completion`, based on `main` at `f37826f`. Being listed under Contributors does not by itself grant write permission.

In the terminal/API attempt on 2026-10-06, the terminal lacked authentication and the GitHub connection returned 403 when creating a blob on the fork. The user then allowed using the signed-in browser to publish the branch/PR. An account's permissions on the repo do not mean the connected app has write access. The current PR status is on GitHub; verification details are in `docs/VERIFICATION.md`.

Do not delete the repo, do not upload the ZIP as a single source file, and do not force-push. Use the current checkout, or unzip the handoff ZIP and copy its contents into a clean clone. The ZIP does not contain `.git` or a Python environment.

```bash
git remote -v
git status --short
git fetch origin
git switch -c codex/crm-completion
.venv-crm/bin/python -m pytest -q
```

If the branch already exists, inspect it and then use `git switch codex/crm-completion`. Do not reset/delete uncommitted CRM changes. If main has moved ahead, compare and resolve conflicts; do not overwrite blindly.

Stage by path; **do not use `git add .`**, to avoid staging out-of-scope files by mistake:

```bash
git add .gitignore AGENTS.md README.md PLAN.md CS582_Group2_Project_Proposal.md requirements.txt
git add src tests scripts notebooks/CRM_Sales_Opportunities.ipynb docs
git add reports/crm/final
git diff --cached --stat
git diff --cached --name-only
```

Read the diff; make sure it contains no out-of-scope data. Then:

```bash
git commit -m "Complete CRM model comparison and reproducible course deliverables"
# Add the remote once; if it already exists, check with git remote -v.
git remote add fork https://github.com/adrewtran/cs582-project.git
git push -u fork codex/crm-completion
gh pr create --repo thai-phan/cs582-project --base main --head adrewtran:codex/crm-completion --title "Complete CRM ML project and reproducible deliverables" --body-file docs/PR_BODY.md
```

Without `gh`: use GitHub **Compare across forks**, choose base `thai-phan:main`, head `adrewtran:codex/crm-completion`, and copy `docs/PR_BODY.md`. Leave it for the team to review; do not merge it yourself. If you have direct push access, you can use a branch on the target repo. If you get a 403, check the account/app permissions; do not change other projects' connections.

If using the handoff patch: apply it to a clean clone at the exact base commit recorded in the handoff. `git apply --check crm-completion.patch`, then `git apply --index crm-completion.patch`. **Do not write `git apply -- --index ...`**: the `--` makes Git treat `--index` as a file name. Do not re-apply the patch to a workspace that already has these changes.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| ModuleNotFoundError | Run setup; use the correct `.venv-crm/bin/python`, not the wrong kernel |
| Python other than 3.12 | Choose a 3.12 runtime; do not loosen the pins and call it verified |
| Missing CSV | Upload the source ZIP that contains `data/crm/`, not just the notebook/results ZIP |
| No GPU | Normal: all models use the CPU |
| Colab quota/disconnect | Save the outputs; rerun when CPU quota is available |
| TabNet best weights warning | Early stopping uses the best checkpoint; the model is not skipped |
| SHAP dependency error | Use the correct env/pins; keep the traceback and manifest for diagnosis |
| AUC around 0.5 | This is the real result; audit for leakage if a score is unusually high |
| Very few High priority deals | Do not force the threshold to look good; the bands are not business-validated |
| Numbers differ from old slides | Use only `reports/crm/final/`; do not mix in old results |
| Slide edits lost after rerun | Edit the generator `src/reporting/deliverables.py`, or keep the hand-edited PowerPoint under a different name |

## 11. Project structure

```
data/crm/                 raw Maven CSVs (read-only); data/crm_simulated/ simulated deals
src/
  data/                   dataset.py (container), crm.py (load/validate/join), split.py (as-of split), preprocess.py
  models/                 zoo.py (5 classifiers), calibration.py (sigmoid), bundle.py (save/load)
  evaluation/             metrics.py, figures.py, audits.py (leakage/priority), novelty_checks.py
  explain/                reference.py (per-deal factors), importance.py (native/permutation/SHAP), priority.py
  reporting/              deliverables.py (paper draft, slides, speaker notes)
  train.py  evaluate.py  predict.py  run_project.py  outputs.py (run folder layout)
scripts/
  tools/                  setup_cpu.py, make_notebook.py, verify_notebook.py, package_handoff.py
  simulation/             make_simulated_*.py, make_industry_trends.py, run_simulated_models.py, run_simulated_seeds.py
  analysis/               expected_revenue_3m.py, product_investment_report.py
reports/crm/final/        data/ models/ metrics/ explain/ checks/ predictions/ figures/ deliverables/ run_manifest.json
tests/  notebooks/  docs/
```
