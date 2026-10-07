## Why

The submitted project is CRM Won/Lost prediction. The previous code had only baseline models, an availability-blind temporal split, and documentation that still pointed toward Lead Scoring. It did not cover the full CRM proposal.

## Changes

- Restore the original CRM proposal and label alternative datasets/older results as historical.
- Validate CRM inputs and join cardinality; keep raw files unchanged.
- Add dated train/validation/test partitions with late-outcome purging and training-only preprocessing.
- Run Dummy prior, LR, RF, MLP and CPU TabNet with validation-only model/threshold selection.
- Evaluate the full metric set, calibration, priority sensitivity and a separate invalid leakage control.
- Export frozen-model open scores, missing-account warnings, reference sensitivities and separate raw RF TreeSHAP diagnostics.
- Add CPU setup, upload-first Colab notebook, detailed README, result-driven paper, slides and three-person ESL notes.

## Evidence and limitations

See `reports/crm/final/run_manifest.json`, test metrics, split manifest and generated deliverables. All five model families ran; the export contains 1,589 Engaging opportunities with 1,088 missing-account warnings. Validation selects LR, whose test AUC is about 0.524. This does not establish useful production prioritization or business uplift.

The dataset has already been inspected, so the holdout is exploratory. Static reference tables, closed-only outcomes, repeated accounts and shared validation duties limit generalization. Priority cutoffs are illustrative; explanations are associative. TabNet may vary across hardware despite the fixed seed.

## Validation

- `.venv-crm/bin/python -m pytest -q` (see `docs/VERIFICATION.md` for final count).
- `.venv-crm/bin/python -m src.run_project` on Linux CPU with Python 3.12.
- Unchanged notebook Python cells executed sequentially using the same CLI. Local Jupyter kernel startup was blocked by socket restrictions; actual Colab login/upload still requires confirmation.
- PPTX/DOCX rendered to PDF for visual checks; actual Google Slides import still requires confirmation.

## Review before merge

- Confirm the diff excludes the unrelated Telco working-file change.
- Run the notebook in the group account and inspect Google Slides import.
- Verify author contributions, presentation timing and course submission requirements.
- Keep final results separate from legacy `reports/crm/temporal/` outputs.

This PR does not deploy a service, submit course work, or claim a production-ready model.
