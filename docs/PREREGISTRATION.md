# Pre-registration: follow-up studies on prediction, learning, thresholds, fair evaluation and pilot design

Written on 2026-10-10, before the full runs of the studies below. It is committed and pushed to PR #4 before those runs, so the git history dates the plan. The code that implements each study is committed with this document and is the exact specification:

| Study | Code | Output folder |
|---|---|---|
| G1 Predictability | `src/evaluation/predictability.py` | `reports/crm/final/analysis/predictability/` |
| G2 Feedback learning (synthetic) | `src/agent/feedback.py`, `src/agent/learning.py` | `reports/crm/final/learning/` |
| G3 Policy thresholds | `src/agent/thresholds.py` | `reports/crm/final/analysis/thresholds/` |
| G4 Equal-information benchmark | `src/agent/benchmark.py` | `reports/crm/final/benchmark/` |
| G5 Pilot power | `src/agent/pilot.py` | `reports/crm/final/pilot/` |

**Disclosure.** During development each study was run in `--quick` mode on small subsets to fix crashes. Those runs used too few deals, seeds or iterations to be informative. Their only checks were mechanics: the incumbent model reproduced its saved validation AUC (0.5691); injected faults were detected; no probability was fabricated; claims verified. No design choice below was changed after those smoke runs. The test period was examined in earlier project iterations (stated in `models/selection_lock.json`), so no test result in this project is pristine.

The frozen model and the existing agent results in `reports/crm/final/` are not refitted or changed by any of these studies.

## G1 Why is real-data prediction weak?

- **Data.** Development rows are labelled deals that closed before the test period starts (`close_date < test_start`). Every analysis and every choice uses only these rows. The test period is scored once, after `analysis/predictability/selection_lock.json` is written.
- **Analyses.**
  - Win rate by split and by month, with a chi-square test of month heterogeneity.
  - Missingness and support of the open pipeline: account missing; open deals older than any closed deal.
  - Win rate against time to close, and the "still open at age a" landmark view.
  - One-feature-at-a-time validation AUC with bootstrap CIs and permutation p-values, Benjamini–Hochberg across features.
  - Whether category win rates repeat from training to validation (Spearman), and whether categories differ more than chance in training (chi-square).
  - Drift: PSI and an adversarial classifier.
  - Label-permutation tests of the production configuration (history/LR, 200 refits) and raw CatBoost (50 refits).
  - Statistical power: Hanley–McNeil.
- **Learnability study.** 4 expanding rolling-origin folds inside the development period, with the same purge rule as `src.data.split`. Nine fixed candidates (`CANDIDATES` in the code).
  - Ranked by mean fold AUC, with a within-fold bootstrap CI and a paired CI against the incumbent (`lr_history`).
  - **Adoption rule:** a candidate replaces the incumbent only if its paired 95% CI against the incumbent is entirely above 0.
  - The locked choice and the best candidate are then scored once on validation and test. The production model is **not** replaced in this PR either way; an adopted candidate would be recommended for a future run.
- **Reformulation.** A landmark ("still open at age a") model, compared using pooled and within-age validation AUC. It is reported as a reformulation of the task, not as an improvement of the engagement-time model.
- **Expected (stated before running).** The study cannot show real signal is absent, only whether it is detectable with these data. Our prior expectation is that no candidate clears the adoption rule.

## G2 Feedback learning

- **Data check.** The CRM tables contain no action history, no reviewer response and no reward. Won/Lost is a deal outcome, not feedback on an agent action, and is not used as a reward.
- **Implemented.**
  - A validated, hash-chained feedback store (`src/agent/feedback.py`), with records that link an agent decision and its propensity to a reviewer's usefulness rating.
  - A synthetic contextual-bandit study (`src/agent/learning.py`) on real decision contexts, with rewards from stated reward models.
- **Method.**
  - Logging: ε-greedy around the fixed policy (ε = 0.2), with propensities recorded.
  - Learner: per-action ridge regression (direct method).
  - Safe-improvement gate: fit on half the feedback; doubly-robust (learned − fixed) estimate on the other half; adopt only if the bootstrap 95% lower bound is > 0.
  - Evaluation: exact expected reward on held-out deals, for 5 feedback sizes × 20 seeds.
- **Environments (fixed before running).**
  - E0: no signal.
  - E1: reviewers agree with the fixed policy.
  - E2: shifted preferences.
- **Hypotheses.**
  - In E0 and E1, the gated learner rarely adopts and never does materially worse than fixed.
  - In E2, adoption and gain grow with the amount of feedback.
  - Ungated learning can do worse than fixed with little feedback.
- **Claim boundary.** No result from G2 is a real-world or business claim. The real agent keeps its fixed policy unless certified human feedback exists (currently 0 records).

## G3 Policy thresholds

- **Separation.** The model classification threshold is already learned on validation and is not changed. Five outcome-relevant policy thresholds are searched on a 2,400-point grid (`GRID` in the code) on validation deals, with outcomes masked while deciding.
- **Objective.** Wilson 95% lower bound of directional accuracy (escalate = predicts a loss, monitor = predicts a win), subject to directional coverage ≥ 5%.
- **Overfitting check.** Tune on the first half of validation by date and compare with the defaults on the second half.
- **Adoption rule.** Adopt only if the paired 95% CI of the accuracy gain on the held-out half is > 0. The test comparison is reported but cannot trigger adoption.
- **Other thresholds.**
  - `min_gain`: chosen by a label-free objective (fewest tool calls with 100% agreement with the full-information decision and complete review-task context).
  - Evidence weights, tool costs and the stale quantile have no label. They stay configurable and are reported as not validatable.
  - Safety rules are outside every search.
- **Expected.** With validation AUC ≈ 0.57, tuned thresholds probably do not generalize.

## G4 Equal-information triage benchmark

- **Task and pool.** Labelled test-period deals, with outcomes masked until every system has finished.
- **Design.** 5 seeds × 240 deals, with exact condition counts per seed:
  - ordinary 120
  - injected missing account 36
  - injected revenue ×1000 error 24
  - transient model failure 24
  - persistent model failure 12
  - external-action request 24
- **Fair conditions.** Same perturbed data and the same retry policy (one retry) for every system. Natural strata of the ordinary cases are reported separately: borderline, contradictory evidence, clear.
- **Systems.**
  - A: prediction only.
  - **A+: equal information** (every agent tool, a fixed order, and the gate: blocking → data fix; |p − t| < 0.05 → review; else label).
  - B: rule assistant.
  - C: agent.
- **Metrics.**
  - Task completion; fabricated probabilities.
  - Injected-issue detection; false blocking alarms.
  - Directional coverage and outcome accuracy, including accuracy versus model-confidence labels at the same coverage.
  - Top-24 review-queue loss rate against the batch loss rate.
  - Claim verification; requests surfaced; external actions executed.
  - Review tasks; tool calls; runtime.
- **Statistics.** 95% bootstrap CIs (deals resampled within seeds), seed ranges, and paired differences C−A, C−A+, C−B, A+−A.
- **Hypotheses, including ones that do not favour the agent.**
  - H1: C and A+ do not differ detectably in directional accuracy.
  - H2: A+ and C detect the injected revenue error; A and B do not. A and B already flag missing accounts.
  - H3: No system's top-24 queue has a loss rate reliably above the batch rate.
  - H4: C uses fewer tool calls than A+.
  - H5: No system executes an external action. A, A+ and B cannot surface a request; that is a capability difference by construction.
  - H6: With the same retry policy, task completion ties.
  - H7: C sends more deals to human review than A+ (a workload cost).
- **Reporting.** Every metric is reported whichever system it favours.

## G5 Pilot

- **Protocol.** `docs/PILOT_PROTOCOL.md` and the forms in `docs/forms/`.
- **Power analysis** (`src/agent/pilot.py`):
  - Closed-form two-proportion sample sizes at α = 0.05 and power 0.80, for absolute uplifts of 2–10 points.
  - The base win rate comes from development data.
  - The cluster design uses the training-period ICC across sales agents.
  - A Monte Carlo check of power, randomizing by agent.
- **Claim boundary.** These are planning numbers only; no uplift is claimed.
