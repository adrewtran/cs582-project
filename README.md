# CS 582 — Predicting CRM Sales Opportunities

## Setup (once)

Python 3.12, CPU only.

```bash
python3.12 scripts/tools/setup_cpu.py      # creates .venv-crm and installs the pinned dependencies
```

All commands below run from the repository root with `.venv-crm/bin/python` (Windows: `.venv-crm\Scripts\python.exe`).

## Live demo of the agent (one command, offline, ~5 s)

```bash
.venv-crm/bin/python -m src.agent_demo                              # one real Engaging deal, narrated step by step
.venv-crm/bin/python -m src.agent_demo --scenario missing-account   # also: borderline, conflict, invalid, tool-failure, prohibited
.venv-crm/bin/python -m src.agent_demo --batch 10 --budget 3        # several deals → prioritized human-review queue
.venv-crm/bin/python -m src.assistant_demo                          # Baseline B: the earlier rule-based assistant
```

The demo loads the saved models in `reports/crm/final/` (no training, no API key, no network) and writes its trace and review tasks to `reports/crm/agent_demo/` (git-ignored). The step-by-step professor demo guide is `docs/EXCELLENT_PROJECT_DEMO_STEP_BY_STEP.md`. The architecture and decision policy are in `docs/AGENT_ARCHITECTURE.md`. The three-speaker ESL transcript is in `docs/ESL_PRESENTATION_TRANSCRIPT.md`.

## Documents

| File | Content |
|---|---|
| `CRM_IEEE_Paper.docx` | Paper (hand-maintained; numbers checked by `tests/test_documents.py`) |
| `CRM_Final.pptx`, `slide.md` | Slides (21 + 3 backup) and their reviewable source with speaker notes |
| `docs/ESL_PRESENTATION_TRANSCRIPT.md` | Three-speaker ESL script and professor Q&A |
| `docs/EXCELLENT_PROJECT_DEMO_STEP_BY_STEP.md` | Step-by-step demo guide (Vietnamese instructions, English sentences) |
| `docs/AGENT_ARCHITECTURE.md` | Agent design, decision policy and novelty hypothesis |
| `NOVELTY_AND_RESULTS.md` | Prediction results, agent comparison, ablations, replay, contribution table |

The notebooks' saved outputs were cleared in the 2026-10-10 merge because the results changed; run them to see current outputs.

## Dataset 1: real CRM data (`data/crm/`)

Results go to `reports/crm/final/`.

### Full pipeline

```bash
.venv-crm/bin/python -m src.run_project                               # train → evaluate → predict → agent evaluation
.venv-crm/bin/python -m src.run_project --quick                       # smoke test → reports/crm/smoke/ (not reportable)
.venv-crm/bin/python -m src.run_project --output reports/crm/my_run   # keep a separate run
```

### Step by step

```bash
.venv-crm/bin/python -m src.train              # 6 models × {raw, history} features; lock selection on validation; calibrate; save to models/
.venv-crm/bin/python -m src.evaluate           # test metrics, 12-row feature-set comparison, figures, SHAP, checks
.venv-crm/bin/python -m src.predict            # score the 1,589 open Engaging deals → predictions/
.venv-crm/bin/python -m src.agent.experiment   # A vs B vs agent, ablations, outcome-masked replay → agent/
```

Add `--output PATH` to `evaluate` and `predict` when `train` used a different output folder.

### Notebooks

Open with the `.venv-crm` kernel and run in order: `notebooks/original/1_train.ipynb`, `2_evaluate.ipynb`, `3_predict.ipynb`.
They run the same steps as the commands above, one cell per step, and write to the same `reports/crm/final/`.

### Score new deals

```bash
.venv-crm/bin/python -m src.predict --input new_deals.csv --save new_deal_scores.csv
```

`new_deals.csv` uses the `sales_pipeline.csv` format: `opportunity_id, sales_agent, product, account, engage_date` (date as `m/d/yy`).

### 3-month expected revenue (illustrative)

```bash
.venv-crm/bin/python scripts/analysis/expected_revenue_3m.py    # → predictions/expected_revenue_3m.json
```

## Dataset 2: simulated data (`data/crm_simulated/`)

Results never go to `reports/crm/final/`.

### Generate the data (only if `data/crm_simulated/` is missing or you want new data)

```bash
.venv-crm/bin/python scripts/simulation/make_simulated_products.py
.venv-crm/bin/python scripts/simulation/make_industry_trends.py
.venv-crm/bin/python scripts/simulation/make_simulated_deals.py --n-deals 100000 --seed 582 --interaction-strength 3
```

### Full pipeline

```bash
.venv-crm/bin/python -m src.train    --data-dir data/crm_simulated      # → reports/crm/simulated/final/
.venv-crm/bin/python -m src.evaluate --output reports/crm/simulated/final
.venv-crm/bin/python -m src.predict  --output reports/crm/simulated/final
```

The simulated data uses 29 features by default: the 18 standard ones plus 11 deal-level ones (sector match, industry trend, rival product). `--quick` on `train` writes to `reports/crm/simulated/smoke/` instead. `evaluate` and `predict` reload the data folder recorded by `train`. On 100,000 deals, `evaluate` is slow because the split-protocol check refits every model.

### Notebooks

Run in order: `notebooks/simulated/1_train.ipynb`, `2_evaluate.ipynb`, `3_predict.ipynb` (writes to `reports/crm/simulated/final/`).
In `2_evaluate.ipynb`, the slow split-protocol and explanation checks run only with `RUN_CHECKS = True`.

### Model comparison: standard vs extended features, and the oracle

```bash
.venv-crm/bin/python scripts/simulation/run_simulated_models.py --quick   # ~20 s → data/crm_simulated/model_results_quick.csv
.venv-crm/bin/python scripts/simulation/run_simulated_models.py           # ~5–6 min → data/crm_simulated/model_results.csv
```

### Repeat over several seeds (error bars)

```bash
.venv-crm/bin/python scripts/simulation/run_simulated_seeds.py --seeds 1 2 3 4 5 --strengths 1 3
.venv-crm/bin/python scripts/simulation/run_simulated_seeds.py --strengths 1 --smooth 1 --tag _smooth
```

Writes `seed_results*.csv` and `seed_summary*.csv` to `data/crm_simulated/`.

### Product ranking (illustrative)

```bash
.venv-crm/bin/python scripts/analysis/product_investment_report.py   # → data/crm_simulated/product_investment.csv
```
