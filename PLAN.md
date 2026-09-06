# CRM Sales Opportunities Project Plan

Status: **CRM-first direction approved by the project owner on 2026-09-06.**

## Scope decision

The Maven CRM Sales Opportunities dataset in `data/crm/` is the primary dataset because it is the dataset named in the submitted proposal and reported to the professor. Earlier experiments that promoted Leads and Bank are not the main project.

## Research question

Can machine-learning models predict whether an opportunity will be Won or Lost using only information available when the deal is engaging?

## Data policy

- Train on 6,711 closed deals: 4,238 Won and 2,473 Lost.
- Do not train on the 2,089 open deals.
- Score the 1,589 Engaging deals after model training as a practical demonstration.
- Never use `deal_stage`, `close_date`, `close_value`, or `opportunity_id` as honest model features.
- Normalize `GTXPro` to `GTX Pro` before joining product data.
- Preserve raw files; generated files go to `reports/crm/`.

## Evaluation

The primary evaluation is a chronological 80/20 holdout ordered by `engage_date`. All preprocessing is learned from the training portion only. A random stratified split may be shown as a secondary sensitivity check.

Models are developed in this order:

1. Dummy majority baseline
2. Logistic Regression
3. Random Forest
4. MLP
5. TabNet

Every model reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix.

## Novel contribution

1. **Leakage audit:** show the misleading result produced by `close_value`, then remove all outcome-time fields.
2. **Fair model comparison:** compare traditional and neural models on the same chronological split.
3. **Explainable open-deal scoring:** export win probabilities and High/Medium/Low priority groups for Engaging deals, followed by global and local explanation work.

An honest ROC-AUC near 0.50 is a valid result: it means the available CRM attributes contain little pre-close signal. The paper must describe this limitation instead of reintroducing leakage.

## Milestones

- **Foundation:** validated CRM loader, chronological split, Dummy/LR/RF, tests, Colab notebook, open-deal CSV.
- **Advanced models:** MLP and TabNet with the same data and split.
- **Explainability:** global importance plus local explanations for selected open deals.
- **Submission:** final tables, figures, paper, slides, and presentation practice.

Detailed design and implementation plan:

- `docs/superpowers/specs/2026-09-06-crm-first-design.md`
- `docs/superpowers/plans/2026-09-06-crm-first-foundation.md`
