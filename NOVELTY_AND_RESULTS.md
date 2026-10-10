# Promised novelty and observed results

The original CRM proposal promises prediction plus model explainability, and a comparison of traditional models with TabNet. It does not promise a new algorithm or a minimum accuracy. The progress slides also proposed win probabilities, follow-up priority and the main factors behind each prediction.

| Promise | Implemented evidence | Interpretation |
|---|---|---|
| Win/loss probabilities and outcome | `src/evaluation/metrics.py`, `src/explain/reference.py`, `reports/crm/final/predictions/open_deal_predictions.csv` | Calibrated P(Won), complementary P(Lost), validation-selected class threshold |
| Why the model gives a score | `src/explain/reference.py`, native/permutation importance outputs | Per-deal positive/negative training-reference sensitivities; associations, not causal advice |
| Explanations describe the model | `src/evaluation/novelty_checks.py`, `checks/explanation_checks.json`, `checks/explanation_agreement_rf.csv` | On the raw RF, reference sensitivity agrees with TreeSHAP (median Spearman 0.76; top-3 overlap 61% vs 17% by chance; leading input has the same sign in 100% of rows). Resetting the top-3 inputs moves the selected model's P(Won) by 0.131 vs 0.042 for 3 random inputs (3.1×; 95% of deals) |
| Feature importance / optional SHAP | `src/explain/importance.py`, `explain/rf_shap_values.csv`, `explain/shap_audit.json` | Additive SHAP for raw RF on 64 test rows; never mislabeled as calibrated LR explanations |
| Traditional models versus TabNet | `src/models/zoo.py`, `metrics/test_metrics.csv` | LR, RF, MLP and TabNet on the same dated split, plus a Dummy control |
| Opportunity priority | `priority`, `checks/priority_test.csv`, validation sensitivity export | High/Medium/Low heuristic; business usefulness is not established |
| Auditability | Data validation, label-availability purge, leakage control, missing-account flags | Helps detect invalid evidence; does not add new predictive information |
| Does the split protocol matter? | `checks/split_protocol_summary.csv`, `checks/split_protocol_comparison.csv` | A random stratified split is up to +0.036 AUC higher (MLP; LR +0.022). Dropping the purge changes AUC on the same test set by −0.029 to +0.006, and every paired bootstrap interval includes 0. All protocols stay near chance |
| Weak data or broken pipeline? | `scripts/simulation/run_simulated_seeds.py`, `data/crm_simulated/seed_summary*.csv`, `SIMULATED_DATA.md` | Same pipeline on simulated deals with a known win formula: TabNet 0.875, MLP 0.873, RF 0.872, LR 0.827, oracle 0.886 (planted interactions, 5 datasets). The models learn when signal exists. The formula is invented, so this says nothing about real sales |

## How strong is each novelty claim?

- **Strongest:** the leakage control (AUC 1.0 vs 0.524), the known-truth synthetic benchmark (models approach the oracle) and the explanation checks (top factors move the score 3× more than random ones).
- **Weaker than expected:** the label-availability split. On this dataset, a naive random split inflates AUC only slightly and does not change the near-chance conclusion. We report it as a measured safeguard, not as the reason the results are honest.
- **Not claimed:** a new algorithm, causal effects, validated priority bands or revenue uplift. The 3-month expected revenue on the final slides is illustrative only.

All numbers here come from one macOS run of `src.run_project` (2026-10-09). An earlier Linux run of the same code gave RF 0.519 and TabNet 0.517 test AUC instead of 0.518 and 0.507; LR, MLP and every conclusion are unchanged.

## Why the model is still weak

The four classifiers have test ROC-AUC between 0.4954 and 0.5238. This is close to 0.5, so their ranking is weak under the tested protocol. It does not prove that every possible model or feature set is incapable of learning this task.

The inputs describe customer, product, agent and timing attributes. The files do not contain richer engagement measures such as response history, call activity, stated budget or competitor status. Their absence is a plausible limitation, not a demonstrated sole cause of the low AUC. The availability-aware split also leaves only 2,975 training rows and removes 1,792 late-label records from earlier periods.

A prior-only classifier gets 60.03% test accuracy by always predicting Won. Thus this accuracy alone is not evidence of useful ranking. A deliberately invalid model using post-outcome `close_value` gets AUC 1.0; it is a leakage control and is excluded from the real predictor.

Sigmoid calibration improves LR test Brier score from 0.2794 to 0.2443, but the Dummy score is still slightly better at 0.2421. Calibration does not improve rank ordering. Explanations reveal the model's behavior; they do not improve accuracy or validate sales recommendations.

Missing account data affects 1,088 of 1,589 open-deal scores and makes that demo less reliable. It is not the explanation for the labeled test AUC: labeled rows have matched accounts.

## Evolution from the illustrative progress slide

The earlier 68%/32% example was illustrative, not a measured result. The earlier fixed 50% class rule was replaced by a validation-selected raw threshold of 0.48, mapped to about 0.6044 after calibration. Final documents and exported metadata use this actual rule. Priority boundaries remain 0.40/0.70 and are explicitly unvalidated.

## Natural ESL answer to the professor

“We don't claim a new algorithm. Our contribution is a CRM workflow that gives a win probability with its reasons, and each safeguard is backed by evidence. A leaky control shows how an outcome field fakes a perfect score. We compared our time-based split with a random split. We tested that the explanations really describe the model. A simulation with a known answer shows the same pipeline learns when there is signal. On the real data, the models are weak, and we report that honestly.”
