# Current CRM results

Mode full. Validation selection: history / logistic_regression.

| experiment | model | validation_roc_auc | test_roc_auc | test_accuracy | test_f1 | test_brier |
| --- | --- | --- | --- | --- | --- | --- |
| raw | dummy_prior | 0.5000 | 0.5000 | 0.6003 | 0.7502 | 0.2421 |
| raw | logistic_regression | 0.5630 | 0.5238 | 0.4372 | 0.2620 | 0.2794 |
| raw | random_forest | 0.5380 | 0.5194 | 0.5900 | 0.7388 | 0.2508 |
| raw | mlp | 0.5286 | 0.4954 | 0.4849 | 0.5135 | 0.3314 |
| raw | tabnet | 0.5205 | 0.4924 | 0.5327 | 0.6555 | 0.2397 |
| raw | catboost | 0.5446 | 0.5424 | 0.6018 | 0.7472 | 0.2429 |
| history | dummy_prior | 0.5000 | 0.5000 | 0.6003 | 0.7502 | 0.2421 |
| history | logistic_regression | 0.5692 | 0.5168 | 0.4262 | 0.2055 | 0.3180 |
| history | random_forest | 0.4957 | 0.4804 | 0.5011 | 0.5683 | 0.2537 |
| history | mlp | 0.5372 | 0.4987 | 0.5011 | 0.5268 | 0.3013 |
| history | tabnet | 0.5289 | 0.4770 | 0.4879 | 0.5313 | 0.2418 |
| history | catboost | 0.5235 | 0.4988 | 0.5261 | 0.6230 | 0.2443 |

Open scores: 1,589; missing-account warnings: 1,088. For Logistic Regression, history has lower test AUC than raw inputs (B minus A -0.0070). This single exploratory comparison does not prove a general predictive benefit. The agent is evidence-linked decision support, not a proven intervention.
