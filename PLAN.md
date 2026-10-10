# CRM completion plan and status

Original CRM proposal is authoritative. Leads, Bank and Telco experiments were removed on 2026-10-07 (recoverable from commit `c5bcfa6`).

## Implemented

1. Raw validation, four-table joins, training EDA and input-coverage audit.
2. Date-group 60%/80% boundaries with train/validation outcome-availability purge.
3. Shared rows/preprocessing contract for Dummy prior, LR, RF, MLP and TabNet on CPU.
4. Validation-only model/threshold choice and frozen-model sigmoid calibration.
5. Full proposal metric set plus AP, Brier/log loss; separate invalid leakage control.
6. Native/permutation importance, RF TreeSHAP reconstruction check, open-deal reference sensitivities and warnings.
7. README; paper and slides are maintained by hand at the repo root (document generator and Colab notebook removed 2026-10-09).

Entry: `python -m src.run_project`. Evidence: `reports/crm/final/`.

## Interpretation

Near-chance AUC does not establish useful sales prioritization. No causal intervention, revenue uplift, or pristine external test is claimed. Open scores are frozen snapshot demonstrations.

## External steps

- Group checks Google Slides import of `CRM_Final.pptx` if slides are presented from Google Slides.
- Review paper, confirm actual contributions, rehearse and verify final slot.
- Authorized integration: push `codex/crm-completion` through the existing writable project fork and open a PR against `thai-phan/cs582-project:main`. No automatic merge.
- Submit course deliverables. No video recording is claimed; guidelines allow slides and/or video.

Detailed records: `docs/COMMITMENT_AUDIT.md`, `docs/superpowers/specs/2026-10-06-crm-completion-design.md`, `docs/superpowers/plans/2026-10-06-crm-completion.md`. These supersede September evaluation/output conventions.
