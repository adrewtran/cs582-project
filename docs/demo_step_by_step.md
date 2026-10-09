# CRM Sales Opportunities: Step-by-Step Demo

Total time is about 10–15 minutes. Results from a full run are already in `reports/crm/final/`, so the demo can use them and skip the long training run. For the talking script, see `docs/demo_script.md`.

## Step 0: Set the story
- **Goal:** predict whether a CRM sales opportunity will be Won or Lost, and explain which factors drive the score.
- **Input:** four CRM tables (pipeline, accounts, products, teams).
- **Output:** a Won/Lost probability and the factors that influence it.
- **Honest framing:** the models are weak (test ROC-AUC about 0.50–0.52). Present this as a rigorous, leakage-free study, not a production model.

## Step 1: Install (once, before the demo) — macOS
No GPU is needed. The tested environment is **Python 3.12 on Linux, CPU only**. macOS has **not** been tested by the team, so run this whole step once before the demo day and fix any problem early. Windows and Linux notes are at the end of this step.

**1a. Install the tools** (Terminal):
```bash
git --version                       # check that git is installed
brew install python@3.12            # needs Homebrew: https://brew.sh
python3.12 --version                # should print Python 3.12.x
```
If `python3.12` is not found after installing, close and reopen Terminal. On Apple Silicon the binary is at `/opt/homebrew/bin/python3.12`, on Intel Macs at `/usr/local/bin/python3.12`. Python 3.13 and 3.14 are rejected by the project, so do not use `python3` if it points to them.

**1b. Get the code:**
```bash
git clone https://github.com/thai-phan/cs582-project.git
cd cs582-project
git checkout main
```
`main` is the default branch. Until PR #7 is merged, `main` does not yet contain the 3-month revenue script and these docs.

**1c. Create the virtual environment:**
```bash
python3.12 -m venv .venv-crm
source .venv-crm/bin/activate       # optional; the prompt now starts with (.venv-crm)
which python                        # with the venv active, should end in .venv-crm/bin/python
```
Creating it by hand is optional: `scripts/setup_cpu.py` creates `.venv-crm` itself if it is missing and reuses it if it exists. Keep the name `.venv-crm`, because every later command calls `.venv-crm/bin/python`. Run `deactivate` to leave the venv. The commands in this guide call the venv's Python directly, so activating it is not required.

**1d. Install the dependencies into the venv:**
```bash
python3.12 scripts/setup_cpu.py     # installs CPU torch first, then requirements.txt, then runs pip check
```
The first install can take several minutes. To start over, delete the `.venv-crm` folder and run steps 1c and 1d again.

**1e. Check that the install worked:**
```bash
.venv-crm/bin/python -c "import sklearn, torch, pytorch_tabnet; print('ok')"
```
If `setup_cpu.py` stops with a version error, you are not using Python 3.12.

**Other systems:**
- **Ubuntu/Debian:** `sudo apt install python3.12 python3.12-venv` (older releases may need the deadsnakes PPA), then the same commands as above.
- **Windows:** install Python 3.12 from <https://www.python.org/downloads/> and tick "Add python.exe to PATH". The interpreter is `.venv-crm\Scripts\python.exe` and activation is `.venv-crm\Scripts\Activate.ps1`. Windows has not been tested.

**No install at all?** Use Google Colab instead: upload `notebooks/CRM_Sales_Opportunities.ipynb`, set the runtime to CPU, then Run all (the README has the full steps).

## Step 2: Verify the environment with a smoke run (about 2 min)
```bash
.venv-crm/bin/python -m src.run_project --quick
```
This writes to `reports/crm/smoke/` and is labelled SMOKE_TEST_NOT_FINAL. Its numbers are not reportable. It only shows that the pipeline runs end to end.

## Step 3: Show the full pipeline (pick one)
- **Live:** `.venv-crm/bin/python -m src.run_project`. This takes several minutes because it trains Dummy, LR, RF, MLP and TabNet on CPU.
- **Faster:** skip the run and open the existing `reports/crm/final/`.
- **Colab:** upload `notebooks/CRM_Sales_Opportunities.ipynb`, set the runtime to CPU, upload the source ZIP and choose Run all. The team's own Colab session has not been verified, so test it once beforehand.

## Step 4: Walk through the results in `reports/crm/final/`
1. `run_manifest.json`: shows `status: complete` and `mode: full`.
2. `data_quality.json` and `leakage_audit.csv`: `close_value`, `close_date`, `deal_stage` and `opportunity_id` are excluded from the predictors.
3. `split_manifest.csv`: the time-based split gives train 2,975, validation 583, test 1,361, with 1,792 rows purged because their outcomes weren't known at the cutoff.
4. `validation_metrics.csv` and `test_metrics.csv`: LR was chosen on validation ROC-AUC. The Dummy model has high accuracy and F1 because it always predicts Won.
5. `calibration_test.csv`: calibration improved Brier from 0.2794 to 0.2443, which is still worse than Dummy at 0.2421.
6. `figures/`: ROC curves and confusion matrices.

## Step 5: Show the explanations
- `feature_importance_*.csv` and `permutation_importance_test.csv`: which features move the score.
- `explanation_reference.json`: sensitivities of the selected model. These are not causes.
- `rf_shap_global.csv` and `shap_audit.json`: RF SHAP as a separate diagnostic, with additivity error under 1e-5.

## Step 6: Show scoring of open deals
Open `open_deal_predictions.csv`. It has 1,589 Engaging rows with `win_probability` and `loss_probability` summing to 1. Rows with a missing account are flagged. Priority is only a heuristic, and open deals are never labelled Lost.

## Step 7: Estimate 3-month expected revenue (illustrative)
```bash
python scripts/expected_revenue_3m.py
```
It multiplies each open deal's `win_probability` by the product's list `sales_price`, then simulates wins and losses 10,000 times. The output goes to `reports/crm/final/expected_revenue_3m.json`, labelled `ILLUSTRATIVE_NOT_A_VALIDATED_FORECAST`.
- **Result:** about 2.50M expected from 1,589 open deals (3.89M if all were won). The simulated range is about 2.38M–2.63M.
- **Account present only:** 501 deals, about 0.83M expected.
- **Why 3 months and not 2 years:** deals close within 138 days (median 45), so the current pipeline covers a short horizon. The data spans only about 10 months and is simulated, so a 2-year forecast is not supported.
- **Say it carefully:** the model is weak (ROC-AUC about 0.5), the range ignores model error, prices are list prices rather than `close_value`, and new deals are not included.

## Step 8: Show the deliverables
`reports/crm/final/deliverables/` holds a PPTX, a DOCX, a Markdown paper and an ESL speaking script.

## Step 9: Run the tests (optional, slow)
```bash
.venv-crm/bin/python -m pytest -q
```
It includes real CPU fits of all five models and a full smoke run. Passing tests do not mean the model is good enough for business use.

## Step 10: Close with the limitations
- The data covers closed deals only, so there is selection bias.
- The validation set was reused.
- The account, team and product tables are static snapshots.
- Accounts repeat across periods.
- The dataset was examined in earlier attempts, so the test set isn't pristine.
- Results don't yet prove the model helps sales prioritisation.

## Tips
- Have `reports/crm/final/` pre-opened in case the live run is slow.
- TabNet results can vary slightly across platforms, so don't promise exact decimals.
- Don't claim Colab or Google Slides were tested unless you've run them.
