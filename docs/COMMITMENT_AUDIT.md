# CRM project commitment audit

## Current upgrade

PR #4 includes six-model raw/history comparison, strict training-only history, evidence-backed assistant, saved-model CLI, 12 slides and three-speaker 11-minute transcript. See `PROFESSOR_DEMO_GUIDE.md` and `PUBLISH_STATUS.md`. Older sections describe the foundation milestone. Cloud rehearsal and submission remain team responsibilities.

## Completion update — 2026-10-06

The sections below preserve the **initial audit**, before implementation. They are not the current missing-work list.
The gaps have now been addressed in the working tree, with evidence under `reports/crm/final/`:

| Initial gap | Implemented evidence |
|---|---|
| Schema/date/ID validation and EDA | `src/datasets/crm.py`, `src/analysis.py`, quality JSON and EDA figures |
| Late outcomes and equal-date leakage | `src/temporal.py`, 2,975/583/1,361 split; 1,792 purged rows |
| Missing MLP / TabNet | `src/models.py`; real CPU fits and five-model comparison |
| Test-driven choice | Validation selection/early stopping/threshold; frozen base + calibration |
| Probability reliability | Brier/log loss, reliability chart and test calibration comparison |
| Priority claims | Test counts/rates and validation cutoff sensitivity; explicitly unvalidated |
| Local explanation consistency | Reference sensitivity export, selected-model metadata, LR reconstruction tests |
| Optional SHAP | Raw RF TreeSHAP on test sample, saved contributions and additivity check |
| Leakage experiment | Separate close_value stump; never used in scoring |
| Cloud-ready reproducibility | CPU setup, upload-first notebook, run manifests and shared CLI |
| Paper, slides, ESL script | Generated editable drafts; 12 slides / 3 speakers |
| Wrong proposal direction | Original CRM DOCX preserved and transcribed; Leads draft archived |

Remaining external steps: the group must verify an actual Colab session/Google Slides import, review contributions and
instructor-specific submission details, rehearse and submit. The user subsequently authorized commit/push/PR; integration uses the existing writable project fork and targets `thai-phan/cs582-project:main`.
Near-chance discrimination remains a research limitation, not an implementation item that can honestly be marked solved.

## Initial audit (historical record)

Audited: 2026-10-06 UTC (2026-10-05 America/Chicago).
Scope: current local working tree on `codex/crm-first-foundation`, not a claim about remote GitHub contents.
No commits or pushes were made during this audit.

## Authority and evidence

- Original uploaded `CS582_Group2_Project_Proposal.docx`: CRM, four model families, six evaluation outputs,
  global explanation, optional SHAP, initial EDA, and reproducible Python programs.
- Uploaded `Project Guidelines_ML_CS_582.pdf`: proposal 10%, slides and/or video 45%, paper 25%, code/scripts 20%.
  Intelligent agents are encouraged, not a mandatory addition to the approved CRM proposal.
- `CS582_CRM_Project_Progress_Final_Flow.pptx`: decision probabilities, outcome and priority, explanation,
  initial priority thresholds to be validated, and an Oct 1–9 finalization milestone.
- `Meeting in _General_ .docx`: reviewed project-related passages; these contain general forum instructions,
  not additional CRM-specific acceptance criteria.
- Local source, notebook, reports, tests, README, PLAN, and both existing CRM design documents.

The original Word proposal and the user's explicit CRM decision take precedence over
`Group2_Project_Proposal.md`, which still describes an abandoned Lead Scoring/Bank direction.
Unseen professor feedback or external edits cannot be verified from these files.

## Commitment matrix

| Commitment | Current evidence | Status / remaining work |
| --- | --- | --- |
| Four-table CRM join, target and feature construction | `src/datasets/crm.py`; 6,711 labeled rows; product correction; no unmatched products | Implemented; strengthen schema/date/stage/ID validation and missing-account reporting |
| Basic exploratory data analysis | Counts and target distribution in progress slides | Partial; no runnable EDA module/notebook with exported distributions, missingness and time/group analysis |
| Logistic Regression and Random Forest | `src/baselines.py`; fresh fit/evaluation succeeded | Implemented, but evaluation limitations below remain |
| MLP | No implementation or results in `src/` / `reports/crm/` | Missing; required by original proposal |
| TabNet | No implementation or results; package not installed | Missing; required by original proposal; do not silently count a skipped model as completed |
| Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | Baseline metrics CSV and PNG figures | Present for Dummy/LR/RF only; missing comparable advanced-model results |
| Global and local explanations | LR coefficients/RF impurity importance; per-deal LR contributions | Partial; no nonlinear local explanations; export model/method identifiers and validate explanation arithmetic |
| SHAP | No code or dependency | Optional in proposal; useful stretch work, not a reason to call the mandatory scope incomplete by itself |
| Win/loss probabilities, outcome, priority | 1,589-row `open_deal_predictions.csv` | Demo implemented; probability reliability and priority validation not implemented |
| Reproducible leakage comparison | Honest features exclude outcome columns | Partial; no executable honest-versus-leaky classification experiment despite PLAN/README claims |
| Free Colab end-to-end workflow | One guided notebook calling baselines and LR scoring | Partial; advanced models and EDA missing; no fresh Colab execution evidence |
| Final presentation and paper | Eight-slide progress deck exists | Progress deck is not a final-results deck; no final paper found in inspected repo/artifact locations |
| Related work and limitations | Three papers listed in original proposal | References exist; no completed related-work synthesis or final discussion found |
| Correct repository documentation | README/PLAN say CRM; proposal Markdown says Leads/Bank | Inconsistent; synchronize without altering the original submitted document |

## Fresh verification

Commands run during this audit:

```bash
.venv/bin/python -m pytest -q
git diff --check
uv pip install --dry-run --python .venv/bin/python -r requirements.txt
```

- 15 tests passed. This verifies existing assertions, not the entire promised project.
- `git diff --check` passed for unstaged changes.
- Dependency dry-run resolved successfully, but did not install or execute the pinned environment.
- The actual verified environment is Python 3.12 with pandas 2.2.3, NumPy 2.3.5,
  scikit-learn 1.8.0, Matplotlib 3.10.8, pytest 9.0.2. These differ from `requirements.txt`.
- Torch, pytorch-tabnet and SHAP are not installed in this environment.
- Existing baseline models were freshly fitted and evaluated without changing report files.

| Existing split / model | Accuracy | F1 (Won) | ROC-AUC |
| --- | ---: | ---: | ---: |
| Dummy majority | 0.6009 | 0.7507 | 0.5000 |
| Logistic Regression | 0.4393 | 0.2725 | 0.5371 |
| Random Forest | 0.5897 | 0.7342 | 0.5199 |

These are exploratory results under the existing protocol. They show weak discrimination for the tested models;
they do not prove that all possible pre-close predictors have no signal.

## Correctness and readiness findings

### 1. The chronological split is not yet a deployment-style backtest

The last training engagement date and first test engagement date are both 2017-09-18.
Of 5,368 training rows, 911 have `close_date >= 2017-09-18`. At the start of that test period,
those final outcomes would not yet have been safely available as training labels.

`close_date` should remain excluded from model inputs but can be used to enforce label availability
when constructing a historical backtest. Entire engagement dates should stay on one side of a boundary.
An inner validation period needs the same safeguards; the current shuffled stratified CV does not emulate
forward-time deployment.

### 2. The notebook chooses the demo model using the hold-out result

It calls LR the strongest model based on test ROC-AUC. A model-selection/calibration/threshold stage should
use training-period validation, not the final test set. Existing hold-out results have already been inspected:
any new report must disclose this history and must not call them a previously unseen confirmatory benchmark.

### 3. Model scores have not been established as reliable business probabilities

LR/RF use balanced class weights. There is no reliability plot, Brier score, calibration comparison, or validation
of the 0.40/0.70 priority boundaries. The current labels are explicit heuristic demo rules, not validated
sales policy. The project should retain that distinction and avoid claiming improved revenue or conversion.

### 4. Open-deal scoring has missing-information and timing limitations

1,088 of the 1,589 scorable Engaging rows lack an account, while labeled rows have matched account data.
The export needs a visible missing-account flag and a limitation explaining that many predictions use
imputed account attributes. Old open deals overlap the training era, so this export is a snapshot demonstration,
not evidence of historic predictions made at each deal's original engagement date.

### 5. Robustness and explanation tests are incomplete

Scoring an empty Engaging set currently raises a zero-samples ValueError.
Existing tests check factor strings but not whether contributions plus the intercept reconstruct the model score.
The explanation is a decomposition relative to transformed zero (numeric centering), not SHAP and not a causal claim.
The CSV currently omits model name, explanation method, decision/priority thresholds, and missing-account metadata.

### 6. Documented outputs exceed implemented outputs

The README describes a leakage demonstration that is not in the notebook. The final-flow presentation remains
a progress deck with work described as upcoming. The final paper, advanced-model figures and related-work
discussion cannot be treated as completed.

## Proposed completion sequence (not implemented in this audit)

1. Freeze the evaluation policy: preserve old numbers as a labeled retrospective baseline; add an as-of-date
   train/validation/test workflow with labels available before each later period, and no test-driven tuning.
2. Add data-quality checks, EDA, and an explicitly segregated leakage experiment. Never reuse the leaky model for scoring.
3. Add CPU-capable MLP and TabNet using shared row splits and train-fitted preprocessing. Record runtime,
   settings, seeds, metrics, confusion matrices and training history. Missing advanced dependencies must be reported explicitly.
4. Validate probability reliability and report threshold/priority sensitivity using validation data only.
   Export both outcomes and reproducible explanations with missing-data warnings. SHAP remains a clearly marked extension.
5. Verify installation and notebook execution in an isolated environment; provide a pinned-ref/ZIP upload path
   while local code is not on GitHub. A local run must not be described as a Google Colab run.
6. Generate a results-based paper draft and final slide deck from verified outputs, including honest limitations,
   references, novelty framed as an applied contribution, and a demo script split across three members.
7. Have the team confirm member contributions, professor-specific feedback and presentation slot before submission.
   No external submission, commit, push or merge is included without an explicit request.

## Methodology references checked

- scikit-learn, Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn, Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- TabNet implementation: https://github.com/dreamquark-ai/tabnet

These support the proposed evaluation/calibration/implementation approach; they do not supply project results.
