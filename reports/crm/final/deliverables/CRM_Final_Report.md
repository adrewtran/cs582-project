# CRM Sales Decision Support

CS 582 Group 2

Hong Thai Phan, Nguyen Khanh An Tran, Hoang Thien Bao Bui

## Abstract

We evaluate whether strictly prior sales history improves CRM win prediction and connect predictions to an inspectable sales assistant. We compare six CPU classifiers on the same availability-aware split, using 18 raw features (A) or 37 raw and historical features (B). Validation selects history / Logistic Regression. Its exploratory test ROC-AUC is 0.5168, compared with 0.5238 for the same model using raw inputs. For Logistic Regression, history has lower test AUC than raw inputs (B minus A -0.0070). This single exploratory comparison does not prove a general predictive benefit. Our contribution is the combination of a leakage audit, a tested temporal feature engine and an evidence-based rule agent. All metrics come from the full run beginning 2026-10-08T09:58:22.178893+00:00.

## 1 Problem and related work

The model estimates P(Won) and its complement among opportunities whose eventual outcome is observed. Open deals are never assigned artificial Lost labels. We do not predict a fixed closing horizon. The intended use is to inspect a score and decide what information needs review, with a person retaining control of sales actions.

Sales win propensity prediction and probability-based B2B workflows already exist [2,3]. TabNet provides an attention-based tabular comparator [4,7]. CatBoost provides a categorical gradient-boosting comparator [8]. We apply existing algorithms rather than claim a novel model architecture. Explainability supports inspection, but SHAP alone is not our novelty.

## 2 Data and preprocessing

The supplied Maven CRM dataset describes a fictitious B2B hardware company [1]. It contains 8,800 opportunities, 85 accounts, 7 products and 35 team records. A fifth CSV is the data dictionary. There are 6,711 closed records with 4,238 Won and 2,473 Lost outcomes. The system scores 1,589 dated Engaging opportunities. It excludes undated Prospecting records. Among scored opportunities, 1,088 lack account information.

We validate key uniqueness, many-to-one joins, stage values, nonnegative numeric fields and valid date ordering. We normalize GTXPro to GTX Pro and the sector spelling technolgy to technology. The source CSV files remain unchanged. run_manifest.json records their SHA-256 hashes and the source hashes for this run.

The 18 raw inputs include product and team information, account attributes, list price and engagement calendar fields. Account revenue is in millions in the source. Revenue per employee converts it to currency units first. Current close_value, close_date, deal_stage and opportunity_id are never model predictors. Stage defines the target; dates enforce availability; IDs link audit records. Histories use other deals' already-observed outcomes, not the current outcome.

One-hot categorical encoders and median imputation fit only training rows. LR, MLP and TabNet scale numerics using training statistics. CatBoost uses native categorical strings and training-fitted numeric medians. Missing categories remain explicit. Static account, product and team snapshots may not describe each entity's true historical state.

## 3 Temporal design and feature engineering

Training has 2,975 rows, validation 583, and test 1,361. We purge 1,792 records whose outcome was not available before the next period. Validation starts 2017-07-15 and test starts 2017-09-18. All six models and both feature variants receive the same row IDs. Win fractions are 64.64%, 60.72% and 60.03%, respectively.

The history engine uses a frozen training archive. At engagement time T, a record can contribute only if close_date < T and its opportunity ID differs from the query. Same-day closures are excluded because intraday order is unknown. Validation and test outcomes never enter the archive, even when they have already closed. This conservative design is an offline comparison, not an online adaptive evaluation.

For each sales agent, account and product, we derive a prior closed count, smoothed win rate, mean observed close value, mean cycle length and cold-start flag. Four global prior statistics complete the 19 added features. The smoothed entity win rate is (prior entity wins + 5 times eligible global win rate) divided by (prior entity count + 5). The smoothing strength is fixed before testing. Missing entity keys use the eligible global prior with local count zero. With no eligible global history, the probability prior is 0.5, count/value/cycle defaults are zero, and cold-start flags remain visible.

The row-level audit records query date, latest eligible closure and archive size. Unit tests check same-day and self exclusion, shuffled input order, future-outcome invariance and held-out-label invariance. The following coverage table counts queries with at least one eligible entity-specific past closure.

| split | entity | rows | with_history | coverage |
| --- | --- | --- | --- | --- |
| train | agent | 2975 | 1951 | 0.6558 |
| train | account | 2975 | 1896 | 0.6373 |
| train | product | 2975 | 1970 | 0.6622 |
| validation | agent | 583 | 583 | 1.0000 |
| validation | account | 583 | 583 | 1.0000 |
| validation | product | 583 | 583 | 1.0000 |
| test | agent | 1361 | 1361 | 1.0000 |
| test | account | 1361 | 1361 | 1.0000 |
| test | product | 1361 | 1361 | 1.0000 |

## 4 Model selection and experimental protocol

A uses raw inputs. B adds histories. We fit Dummy prior, Logistic Regression, Random Forest, MLP, TabNet and CatBoost for both variants. Dummy returns the training win fraction. LR uses L2 regularization and C=1. RF uses 300 trees and minimum leaf size 5. MLP uses hidden layers 64 and 32, alpha 0.001, learning rate 0.001, at most 120 epochs and patience 15. TabNet uses n_d=n_a=8, three steps, learning rate 0.02, at most 80 epochs and patience 12. CatBoost uses depth 5, learning rate 0.05, up to 500 iterations, L2 leaf regularization 5 and validation early stopping 40. CPU threads are limited and the random seed is 42. Smoke runs use smaller budgets and are explicitly unreportable.

Validation ROC-AUC selects the model and experiment, with lower validation Brier score and deterministic name ordering breaking ties. We retain validation-selected neural checkpoints. A validation macro-F1 search chooses class thresholds. Positive-slope sigmoid calibration uses validation and keeps the base classifier frozen. selection_lock.json saves both experiments' choices before either evaluates test predictions. The validation sample serves multiple roles, so these estimates can be optimistic. Test results do not trigger another configuration search.

Diagnostic C is a separate depth-one tree using the current final close value. It illustrates invalid outcome leakage and never supplies real opportunity scores. We report AP as average precision, rather than calling it trapezoidal PR-AUC. Precision, recall and F1 treat Won as positive. Confusion counts and additional metrics appear in the appendix and CSV files.

## 5 Validation and test results

Validation comparisons determine selection. These numbers are tuning diagnostics rather than an independent estimate of generalization.

| experiment | model | validation_roc_auc | validation_brier | validation_threshold |
| --- | --- | --- | --- | --- |
| raw | dummy_prior | 0.5000 | 0.2400 | 0.5000 |
| raw | logistic_regression | 0.5630 | 0.2514 | 0.4800 |
| raw | random_forest | 0.5380 | 0.2429 | 0.5400 |
| raw | mlp | 0.5286 | 0.2911 | 0.4600 |
| raw | tabnet | 0.5205 | 0.2386 | 0.5900 |
| raw | catboost | 0.5446 | 0.2386 | 0.5800 |
| history | dummy_prior | 0.5000 | 0.2400 | 0.5000 |
| history | logistic_regression | 0.5692 | 0.2755 | 0.4100 |
| history | random_forest | 0.4957 | 0.2573 | 0.4900 |
| history | mlp | 0.5372 | 0.2732 | 0.4200 |
| history | tabnet | 0.5289 | 0.2383 | 0.6200 |
| history | catboost | 0.5235 | 0.2442 | 0.5300 |

The test comparison below uses the original probability scale of each model and its validation-selected threshold. Lower Brier and log loss are better. AUC 0.5 represents no useful ranking.

| experiment | model | test_accuracy | test_f1 | test_roc_auc | test_brier |
| --- | --- | --- | --- | --- | --- |
| raw | dummy_prior | 0.6003 | 0.7502 | 0.5000 | 0.2421 |
| raw | logistic_regression | 0.4372 | 0.2620 | 0.5238 | 0.2794 |
| raw | random_forest | 0.5900 | 0.7388 | 0.5194 | 0.2508 |
| raw | mlp | 0.4849 | 0.5135 | 0.4954 | 0.3314 |
| raw | tabnet | 0.5327 | 0.6555 | 0.4924 | 0.2397 |
| raw | catboost | 0.6018 | 0.7472 | 0.5424 | 0.2429 |
| history | dummy_prior | 0.6003 | 0.7502 | 0.5000 | 0.2421 |
| history | logistic_regression | 0.4262 | 0.2055 | 0.5168 | 0.3180 |
| history | random_forest | 0.5011 | 0.5683 | 0.4804 | 0.2537 |
| history | mlp | 0.5011 | 0.5268 | 0.4987 | 0.3013 |
| history | tabnet | 0.4879 | 0.5313 | 0.4770 | 0.2418 |
| history | catboost | 0.5261 | 0.6230 | 0.4988 | 0.2443 |

The selected model changes from raw-feature AUC 0.5238 to selected-variant AUC 0.5168, a descriptive difference of -0.0070. For Logistic Regression, history has lower test AUC than raw inputs (B minus A -0.0070). This single exploratory comparison does not prove a general predictive benefit. The history representation may capture time trends and sparsity rather than stable opportunity information. Training history grows over time, while validation/test use a frozen archive; this changes the feature distribution. Those are plausible explanations, not separately verified causes. We did not run a multi-seed or cluster-bootstrap significance study.

The highest test AUC is a descriptive observation only. We retain the validation winner rather than choose a model after inspecting test results. Dummy can attain a high Won-class F1 by predicting the majority class, so F1 alone is not enough to demonstrate useful prioritization.

### Probability calibration

| variant | roc_auc | brier | log_loss | threshold |
| --- | --- | --- | --- | --- |
| raw | 0.5168 | 0.3180 | 0.8405 | 0.4100 |
| calibrated | 0.5168 | 0.2489 | 0.6910 | 0.6109 |

Calibration changes the probability scale while preserving ranking, apart from numerical ties. Its result does not establish reliable future probabilities. Brier must be interpreted against the prior-only baseline as well as the uncalibrated selected model.

## 6 Explanations and sales assistant

The selected calibrated model produces the win probability and complementary loss probability. Its local explanations replace one feature with a training median or mode and measure the probability difference. These are nonadditive reference sensitivities, not causal effects. Correlated history features can yield unrealistic single-feature replacements. RF TreeSHAP is a separate supporting diagnostic for the uncalibrated forest, with numerical reconstruction checks [6]. It is not mislabeled as an explanation of another selected classifier.

The deterministic assistant receives the probability, historical context, signed sensitivities and warnings. It returns loss risk HIGH below P(win)=0.40, MEDIUM from 0.40 to below 0.70, and LOW at 0.70 or above. Risk describes estimated loss probability; it is distinct from the legacy High win-priority label. Fixed bands are illustrative, not a validated allocation policy.

Rules request account completion when identity is missing, direct verification when fewer than five account outcomes exist, product-fit review when at least five past product outcomes lag the eligible global win rate by over 0.10, or a second sales review when high loss risk and sufficiently supported agent history agree. The output includes at most two triggered rules plus two universal checks: verify current opportunity status and review the evidence before action. Each action exposes its rule ID, observed evidence and rationale. No rule sends a message, changes a price, measures uplift, or claims an intervention will cause a win. No paid LLM API is required.

Our course-level contribution has three linked parts: an explicit leakage audit, an availability-tested historical feature engine and an inspectable rule-based sales assistant. CatBoost is a stronger comparator and SHAP is supporting analysis. Neither is claimed as our own new algorithm. The agent is decision support, not autonomous planning or a learned sales policy.

## 7 Live demonstration

The default example is opportunity 01XZ9CRY, account Initech, product GTX Plus Pro. The saved model estimates win probability 53.43% and loss probability 46.57%, with loss risk MEDIUM. Its real outcome remains unknown. Scoring context: post_model_engagement_snapshot. The example is selected by date and ID, not a high score.

The CLI reloads model_bundle.joblib, reconstructs histories from its saved archive, recalculates the probability and reasons, and runs the rule engine. Tests compare it with the exported score. Records whose engagement predates model availability are explicitly marked retrospective snapshots. Even later records remain historical demonstrations, not a live CRM deployment. Only load joblib bundles from trusted sources.

## 8 Limitations and next evidence

The data has been inspected in earlier project iterations, so this is an exploratory holdout. Closed-only sampling and end-of-snapshot censoring remain. Accounts and agents recur across periods, so the experiment is not a test of entirely new customers. Input tables are static. The training archive remains frozen and early training examples have thin or no history. Open records with missing accounts differ from the complete-account labeled sample.

The measured discrimination is weak. Recommendations have logical rule tests but no user study or intervention-outcome evaluation. The rule thresholds and probability bands need prospective validation. We do not claim increased revenue, expected deal value, causal actions or production readiness. Future work should collect dated interaction, qualification and competitor information; evaluate rolling periods or an external cohort; reserve separate calibration data; and test whether users find the evidence-linked actions useful. Changes should be predeclared before consulting a new test.

## 9 Reproducibility and conclusion

Run python scripts/setup_cpu.py with Python 3.12, then .venv-crm/bin/python -m src.run_project. The CPU pipeline writes both experiments, the selection lock, all metrics, audits, models, 1-row live demo example and batch assistant outputs. The notebook invokes the same code. A successful software run is distinct from a useful predictive model. Our verified execution environment is development CPU; actual group Colab execution, Google Slides import and course submission require team confirmation.

The extension strengthens the experimental audit and connects model output to transparent review actions. It does not justify a claim that historical features improve predictive accuracy on this dataset. Reporting this distinction is central to the project.

## Appendix Complete metrics and confusion counts


### Raw features

| model | precision | recall | macro_f1 | balanced_accuracy |
| --- | --- | --- | --- | --- |
| Dummy prior | 0.6003 | 1.0000 | 0.3751 | 0.5000 |
| Logistic Regression | 0.6154 | 0.1665 | 0.4036 | 0.5051 |
| Random Forest | 0.5982 | 0.9657 | 0.3933 | 0.4957 |
| MLP | 0.5929 | 0.4529 | 0.4832 | 0.4930 |
| TabNet | 0.5879 | 0.7405 | 0.4647 | 0.4806 |
| CatBoost | 0.6036 | 0.9804 | 0.4047 | 0.5068 |

### Raw features

| model | average_precision | log_loss | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy prior | 0.6003 | 0.6775 | 0 | 544 | 0 | 817 |
| Logistic Regression | 0.6138 | 0.7541 | 459 | 85 | 681 | 136 |
| Random Forest | 0.6169 | 0.6990 | 14 | 530 | 28 | 789 |
| MLP | 0.5879 | 0.9353 | 290 | 254 | 447 | 370 |
| TabNet | 0.6181 | 0.6723 | 120 | 424 | 212 | 605 |
| CatBoost | 0.6381 | 0.6793 | 18 | 526 | 16 | 801 |

### History features

| model | precision | recall | macro_f1 | balanced_accuracy |
| --- | --- | --- | --- | --- |
| Dummy prior | 0.6003 | 1.0000 | 0.3751 | 0.5000 |
| Logistic Regression | 0.6084 | 0.1236 | 0.3782 | 0.5021 |
| Random Forest | 0.5913 | 0.5471 | 0.4887 | 0.4896 |
| MLP | 0.6117 | 0.4627 | 0.4996 | 0.5107 |
| TabNet | 0.5896 | 0.4835 | 0.4834 | 0.4890 |
| CatBoost | 0.5962 | 0.6524 | 0.4925 | 0.4944 |

### History features

| model | average_precision | log_loss | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy prior | 0.6003 | 0.6775 | 0 | 544 | 0 | 817 |
| Logistic Regression | 0.6095 | 0.8405 | 479 | 65 | 716 | 101 |
| Random Forest | 0.5895 | 0.7005 | 235 | 309 | 370 | 447 |
| MLP | 0.5993 | 0.8145 | 304 | 240 | 439 | 378 |
| TabNet | 0.5803 | 0.6769 | 269 | 275 | 422 | 395 |
| CatBoost | 0.5999 | 0.6818 | 183 | 361 | 284 | 533 |

## References

[1] Maven Analytics. CRM Sales Opportunities. Fictitious B2B computer-hardware company; four data tables. https://mavenanalytics.io/data-playground/crm-sales-opportunities (accessed 2026-10-06). Counts in this report are computed from the supplied CSV files, not website metadata.

[2] J. Yan, M. Gong, C. Sun, J. Huang, and S. M. Chu. Sales Pipeline Win Propensity Prediction: A Regression Approach. IFIP/IEEE IM, 2015. https://arxiv.org/abs/1502.06229

[3] A. Rezazadeh. A Generalized Flow for B2B Sales Predictive Modeling: An Azure Machine Learning Approach. Forecasting, 2020. https://doi.org/10.3390/forecast2030015 ; author manuscript: https://arxiv.org/abs/2002.01441

[4] S. O. Arik and T. Pfister. TabNet: Attentive Interpretable Tabular Learning. AAAI 35(8), 6679–6687, 2021. https://doi.org/10.1609/aaai.v35i8.16826

[5] scikit-learn. Probability calibration. https://scikit-learn.org/stable/modules/calibration.html (accessed 2026-10-06).

[6] SHAP documentation. TreeExplainer. https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html (accessed 2026-10-06).

[7] DreamQuark. pytorch-tabnet implementation and documentation. https://github.com/dreamquark-ai/tabnet (version 4.1.0 used).

[8] L. Prokhorenkova et al. CatBoost: unbiased boosting with categorical features. NeurIPS, 2018. https://arxiv.org/abs/1706.09516 ; implementation https://catboost.ai/docs/ (version 1.2.10 used).