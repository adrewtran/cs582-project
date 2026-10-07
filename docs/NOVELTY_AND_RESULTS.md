# Promised novelty and observed results

The original CRM proposal promises prediction plus model explainability, and a comparison of traditional models with TabNet. It does not promise a new algorithm or a minimum accuracy. The progress slides also proposed win probabilities, follow-up priority and the main factors behind each prediction.

| Promise | Implemented evidence | Interpretation |
|---|---|---|
| Win/loss probabilities and outcome | `src/evaluation.py`, `src/explain.py`, `reports/crm/final/open_deal_predictions.csv` | Calibrated P(Won), complementary P(Lost), validation-selected class threshold |
| Why the model gives a score | `src/explain.py`, native/permutation importance outputs | Per-deal positive/negative training-reference sensitivities; associations, not causal advice |
| Feature importance / optional SHAP | `src/analysis.py`, `rf_shap_values.csv`, `shap_audit.json` | Additive SHAP for raw RF on 64 test rows; never mislabeled as calibrated LR explanations |
| Traditional models versus TabNet | `src/models.py`, `test_metrics.csv` | LR, RF, MLP and TabNet on the same dated split, plus a Dummy control |
| Opportunity priority | `priority`, `priority_test.csv`, validation sensitivity export | High/Medium/Low heuristic; business usefulness is not established |
| Auditability | Data validation, label-availability purge, leakage control, missing-account flags | Helps detect invalid evidence; does not add new predictive information |

## Why the model is still weak

The four classifiers have test ROC-AUC between 0.4954 and 0.5238. This is close to 0.5, so their ranking is weak under the tested protocol. It does not prove that every possible model or feature set is incapable of learning this task.

The inputs describe customer, product, agent and timing attributes. The files do not contain richer engagement measures such as response history, call activity, stated budget or competitor status. Their absence is a plausible limitation, not a demonstrated sole cause of the low AUC. The availability-aware split also leaves only 2,975 training rows and removes 1,792 late-label records from earlier periods.

A prior-only classifier gets 60.03% test accuracy by always predicting Won. Thus this accuracy alone is not evidence of useful ranking. A deliberately invalid model using post-outcome `close_value` gets AUC 1.0; it is a leakage control and is excluded from the real predictor.

Sigmoid calibration improves LR test Brier score from 0.2794 to 0.2443, but the Dummy score is still slightly better at 0.2421. Calibration does not improve rank ordering. Explanations reveal the model's behavior; they do not improve accuracy or validate sales recommendations.

Missing account data affects 1,088 of 1,589 open-deal scores and makes that demo less reliable. It is not the explanation for the labeled test AUC: labeled rows have matched accounts.

## Evolution from the illustrative progress slide

The earlier 68%/32% example was illustrative, not a measured result. The earlier fixed 50% class rule was replaced by a validation-selected raw threshold of 0.48, mapped to about 0.6044 after calibration. Final documents and exported metadata use this actual rule. Priority boundaries remain 0.40/0.70 and are explicitly unvalidated.

## Natural ESL answer to the professor

“Our contribution is to combine CRM prediction with explanations and a careful evaluation process. We estimate a win probability, show the main factors behind the score, and compare traditional models with TabNet. The current models have weak predictive performance on this dataset. We report that limitation clearly instead of using information that is only available after a deal closes. The explanation helps us understand the model, but it does not automatically make the model more accurate.”
