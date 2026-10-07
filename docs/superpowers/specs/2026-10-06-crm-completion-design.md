# CRM completion design

User approved completion end-to-end in one pass, including the as-of-date correction, on 2026-10-05 Chicago time.
The original CRM Word proposal is authoritative. No change to raw data, no paid services, no push/PR yet.

## Evaluation

Keep the old baseline module/results as explicitly retrospective. Add a primary dated protocol using engagement-date
boundaries near the 60th and 80th row percentiles, keeping equal dates together. Base training rows must engage and
close before validation starts. Validation rows engage in the middle period and must close before test starts.
Test rows engage in the final period. Drop/purge unobservable labels, record every row's role and purge reason.
No class labels or outcome-time attributes enter model features. All five models use identical row indices and
training-fitted preprocessing. Historical hold-out numbers were already seen; disclose that this is exploratory.

Fit Dummy-prior, LR, RF, sklearn MLP and PyTorch TabNet on CPU. The primary LR/RF models use unweighted classes to
avoid deliberately changing class priors; legacy balanced models remain a separate comparison. Fix seed 42 and
bounded training budgets. Use validation for neural early stopping, model selection (ROC-AUC, then Brier, then name),
and decision threshold (macro-F1, ties closest to 0.5). Test metrics never select settings.

Fit a sigmoid calibration map on the selected frozen model's validation probabilities, then evaluate raw and
calibrated probabilities on test. Do not report calibration-fit validation scores as out-of-sample evidence.
The selected model is not retrained after calibration: this preserves the frozen score/calibration relationship.
Classify at the selected raw-score threshold mapped through the calibration function. Priority boundaries stay
explicitly heuristic 0.40/0.70 and are assessed descriptively on held-out data; no revenue uplift claim.

## Outputs and explanations

Generate EDA, model metrics/ROC/PR/confusion/calibration figures, settings, timings, learning curves, split manifest,
raw-file SHA256 hashes, environment versions, a leakage comparison with a clearly disallowed close_value model,
selected-model permutation importance and per-opportunity reference-replacement sensitivity explanations.
These sensitivity differences are not SHAP, not additive, and not causal. Keep the existing exact LR contribution
demo as an explicitly separate method. If SHAP installs, generate RF TreeSHAP for a bounded held-out sample and
check additivity; any omission must appear in run metadata.

Score the 1,589 Engaging opportunities as a frozen-model snapshot demo, never as a historical backtest.
Export win/loss probabilities, Won/Lost, fixed-policy priority, model/method/threshold metadata, account-missing flags,
top positive/negative reference sensitivities. Empty input produces a schema-correct empty table. All inputs are
validated; no silent unknown-stage/date/duplicate-ID removal. Prospecting is not scored.

## Delivery

One command `python -m src.run_project` executes audit, EDA, all models, evaluation, explanations and document
generation. `--quick` is explicitly smoke testing, not the final experiment. No model silently disappears.
A Colab notebook supports a local extracted ZIP first and pinned GitHub revision after push, checks subprocess
failures, executes tests and the same entry point, and offers output download.
Create an English paper draft, editable PowerPoint final deck, 3-member natural ESL speaker script and detailed
Vietnamese/English README from actual generated results. Bibliography must use verified sources. Team must review
authorship, contributions, citations and professor feedback before submission; no fabricated presentation/video.

## Verification and boundaries

Use a clean Python 3.12 environment, CPU-only torch, unit/integration tests, full execution and notebook execution
locally. Do not describe local validation as Google Colab validation. Preserve existing unrelated Telco changes.
Work in the existing dedicated feature branch as requested. No commits until explicitly requested; no automatic
push/merge. Preserve full handoff ZIP and a patch excluding Telco and environments.
