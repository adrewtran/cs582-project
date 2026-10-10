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
8. PR #4 merged with main (2026-10-10): CatBoost comparator; strictly prior history features (`src/data/history.py`) compared with raw features under one validation selection lock.
9. Evidence-grounded decision agent (`src/agent/`): value-of-information tool planning, reliability-gated decisions, safety guard with approval queue, replayable traces, batch review queue; evaluated against prediction-only and rule-based baselines with ablations and an outcome-masked replay (`reports/crm/final/agent/`). Demo: `python -m src.agent_demo`; guide: `docs/EXCELLENT_PROJECT_DEMO_STEP_BY_STEP.md`.

Entry: `python -m src.run_project`. Evidence: `reports/crm/final/`. Agent design: `docs/AGENT_ARCHITECTURE.md`.

## Interpretation

Near-chance AUC does not establish useful sales prioritization. No causal intervention, revenue uplift, or pristine external test is claimed. Agent metrics describe workflow quality (evidence, safety, traceability), not sales outcomes. Open scores are frozen snapshot demonstrations.

## External steps

- Group checks Google Slides import of `CRM_Final.pptx` if slides are presented from Google Slides.
- Review paper, confirm actual contributions, rehearse and verify final slot.
- Authorized integration: PR #4 (`adrewtran:codex/crm-history-agent` → `thai-phan/cs582-project:main`). No automatic merge.
- Submit course deliverables. No video recording is claimed; guidelines allow slides and/or video.

The former `docs/` folder (commitment audit, design specs) was removed on 2026-10-10 and is recoverable from git history; `docs/` now holds only the agent architecture, the professor demo guide and the ESL transcript.
