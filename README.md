# Predicting CRM Sales Opportunities Using Machine Learning

CS 582 Group 2 project. The goal is to predict whether a CRM sales opportunity will be **Won** or **Lost** using information available when the opportunity is engaging.

## Run free in Google Colab

[Open CRM notebook in Colab](https://colab.research.google.com/github/thai-phan/cs582-project/blob/main/notebooks/CRM_Sales_Opportunities.ipynb)

The notebook clones this repository, installs the required packages, validates the four CRM tables, trains the baseline models, and exports predictions for open Engaging deals.

## Command-line verification

```bash
git clone https://github.com/thai-phan/cs582-project.git
cd cs582-project
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.datasets crm
python -m src.baselines crm --split temporal
```

Generated baseline files are written under `reports/crm/`.

## Honest feature policy

The honest model excludes `deal_stage`, `close_date`, `close_value`, and `opportunity_id`. These fields reveal the outcome or do not generalize. The project separately demonstrates how leakage can create an unrealistic result.

## Current milestone

- CRM data preparation and joins
- Chronological train/test split
- Dummy, Logistic Regression, and Random Forest baselines
- Open Engaging-deal probability and priority export
- MLP, TabNet, and richer explanations are the next milestone

See `PLAN.md` for the approved direction.
