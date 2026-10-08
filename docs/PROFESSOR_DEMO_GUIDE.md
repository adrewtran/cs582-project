# Professor demo — CRM Sales Opportunities

Target: **11 minutes, including the live demo** (allow 10–12 minutes). The speaking assignments below do not claim development authorship.

| Speaker | Slides | Time | Topic |
|---|---|---|---|
| Hong Thai Phan | 1–4 | 0:00–3:40 | Problem, inputs, time split and history |
| Nguyen Khanh An Tran | 5–8 | 3:40–7:20 | Models, results, leakage audit and assistant |
| Hoang Thien Bao Bui | 9–12 | 7:20–11:00 | Live demo, explanations, novelty and limits |

The exact natural spoken English and Q&A are in `reports/crm/final/deliverables/SPEAKER_SCRIPT_ESL.docx` and `.md`. The PowerPoint includes speaker notes. Aim for 100–115 words per minute and rehearse with a timer.

## 1. Download the correct source

Before PR #4 is merged, use **adrewtran/cs582-project**, branch **codex/crm-history-agent**. After merge, use **thai-phan/cs582-project**, branch **main**.

1. Select the branch, then **Code → Download ZIP**. Keep this ZIP compressed.
2. Download `notebooks/CRM_Sales_Opportunities.ipynb` from the same branch using **Download raw file**.
3. Do not upload an old ZIP, a patch, or a results-only ZIP. Source must contain `src/run_project.py`, `assets/CRM_History_Agent_Template.pptx` and `data/crm/sales_pipeline.csv`.

The main experiment uses the original CRM files. Separate synthetic experiments in `data/crm_synthetic` are preserved, not mixed into these results.

## 2. Prepare Colab before presentation day

1. Open https://colab.research.google.com/ and sign in.
2. Choose **File → Upload notebook**, selecting the downloaded notebook.
3. The tested environment requires Python **3.12**, accelerator **None/CPU**. Setup stops on unsupported Python versions. Check runtime availability in your account.
4. Choose **Runtime → Run all**.
5. Upload exactly one source ZIP when prompted.
6. Wait for dependency setup, regression checks, six models on two feature sets, document generation and the saved-model demo. Installation speed and runtime quota vary; do this before the talk.
7. Confirm `PASS: full`, 12 A/B rows and `CRM SALES ASSISTANT` output.
8. Save the result ZIP downloaded by the last cell before ending the temporary runtime.

No paid API or GitHub write credential is required. Colab login/upload/download and Google Slides import still need team rehearsal; local execution is not proof of cloud UI compatibility.

## 3. Prepare one short live cell

After Run all, `ROOT`, `PYTHON` and `OUTPUT` already exist. Add this code cell:

```python
import json, subprocess
result = subprocess.run(
    [str(PYTHON), '-m', 'src.demo', '--bundle', str(OUTPUT / 'model_bundle.joblib'), '--json'],
    cwd=ROOT, check=True, capture_output=True, text=True)
demo = json.loads(result.stdout)
p = demo['model_prediction']
print('Opportunity:', demo['opportunity'])
print(f"Win: {p['win_probability']:.1%}  Loss: {p['loss_probability']:.1%}")
print('Predicted outcome:', p['predicted_outcome'])
print('Classification threshold:', round(p['decision_threshold'], 3))
print('Loss risk:', demo['agent_recommendation']['loss_risk'])
print('\nReasons:')
for factor in p['key_factors']:
    print(factor)
print('\nActions and evidence:')
for action in demo['agent_recommendation']['actions']:
    print(action)
print('\nTiming:', demo['timing'])
```

This reloads the saved model and recomputes the score, without training. The normal example is **01XZ9CRY / Initech / GTX Plus Pro**, approximately **53.4% win, 46.6% loss**, **MEDIUM loss risk**. Its actual outcome is unknown. The default example is selected by date and ID, not its score. A win probability above 50% can have a Lost label because validation selected a classification threshold of about 61.1%. Risk bands are separate heuristics.

## 4. Bao's demonstration on slide 9 (60–90 seconds)

1. Switch to the prepared Colab cell: “Let me show one real record from our dataset. This command loads our saved model and calculates the score again.”
2. Run it and point to ID and probabilities: “This opportunity has about a fifty-three percent estimated chance of winning. Its actual result is still unknown.”
3. Point to one reason: “This shows how the model score changes when we replace one input with a training reference. It describes the model's behavior.”
4. Show one action and evidence: “The assistant asks us to verify the current deal status. The record is Engaging, but our dataset is a historical snapshot.”
5. Return to slide 10: “A salesperson still needs to review this information. We have not shown that these suggestions increase sales.”

Do not run setup or training during the talk. Do not scroll through all 1,589 records. If loading takes longer than about 15 seconds, use the saved example and explain that it is saved output, not a fresh calculation.

## 5. Optional missing-account example

Only show this if asked. An earlier engagement can carry a retrospective warning.

```python
import csv
with (OUTPUT / 'open_deal_predictions.csv').open() as handle:
    row = next((r for r in csv.DictReader(handle) if r['account_missing'] == 'True'), None)
if row:
    subprocess.run([str(PYTHON), '-m', 'src.demo', '--bundle', str(OUTPUT / 'model_bundle.joblib'),
                    '--opportunity-id', row['opportunity_id']], cwd=ROOT, check=True)
else:
    print('No missing-account example in this run.')
```

Say: “Here the account field is missing. The assistant asks us to complete it and shows the missing field as evidence. This does not prove that completing the field will win the deal.”

## 6. Result interpretation and backup

The comparison contains 12 rows and the selected-model table contains six. The supplied data has 1,589 scored Engaging opportunities, including 1,088 missing accounts. Won/Lost probabilities sum to one. Validation selects history / Logistic Regression; its test AUC is about .5168 versus .5238 for raw LR. History did not improve this pair. Passing tests proves software consistency, not useful sales prediction.

Keep the slides, prepared demo cell, `demo_example.json`, `ablation_comparison.csv` and `deliverables/RESULTS_SUMMARY.md` open. If Colab disconnects, say: “This is the saved output from our completed run. The live runtime is unavailable right now.” If there are zero eligible rows, do not invent a probability; the notebook skips the demo and still downloads results.

Upload the PPTX to Drive and open it with Google Slides. Check all 12 slides, the comparison table and speaker notes. Rehearse the transitions between speakers and back to slide 10. Review names, actual team contributions and the professor's final slide limit before submission.
