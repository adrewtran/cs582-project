# Results from the current run

Mode: **full**. Selected on validation: **Logistic Regression**.

| model               |   accuracy |     f1 |   roc_auc |   brier |
|:--------------------|-----------:|-------:|----------:|--------:|
| dummy_prior         |     0.6003 | 0.7502 |    0.5000 |  0.2421 |
| logistic_regression |     0.4372 | 0.2620 |    0.5238 |  0.2794 |
| random_forest       |     0.5900 | 0.7388 |    0.5194 |  0.2508 |
| mlp                 |     0.4849 | 0.5135 |    0.4954 |  0.3314 |
| tabnet              |     0.5628 | 0.6685 |    0.5173 |  0.2468 |

All five models ran. Open scores: 1,589; missing-account warnings: 1,088. Near-chance AUC does not support production prioritization. Calibration and priority bands are diagnostics, not proven business improvement. See the paper for limitations.
