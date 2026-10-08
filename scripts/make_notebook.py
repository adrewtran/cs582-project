"""Generate the self-contained upload-first Colab entry notebook."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def create_notebook():
    nb=nbf.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':'CRM_Sales_Opportunities.ipynb'},'language_info':{'name':'python','version':'3.12'}}
    nb.cells=[
        nbf.v4.new_markdown_cell('''# CS 582 — CRM Sales Opportunities\n
**Run all**: upload the project ZIP, install CPU dependencies, run all six models, inspect results, download outputs. No GitHub write permission, API key, paid runtime, or GPU is needed. Setup requires internet and available Colab CPU quota. Use a Python 3.12 runtime; other Python versions are not claimed as tested.\n
Đây là notebook chạy toàn bộ project. Bấm **Runtime → Run all**, chọn ZIP được giao (không upload patch). Cell cài đặt có thể mất vài phút. Kết quả yếu vẫn là kết quả hợp lệ; không dùng `close_value` làm predictor.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import os, sys, subprocess, tempfile, zipfile
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

configured = os.environ.get("CRM_PROJECT_ROOT")
if configured:
    ROOT = Path(configured).resolve()
elif Path.cwd().joinpath("src/run_project.py").exists():
    ROOT = Path.cwd()
elif Path.cwd().parent.joinpath("src/run_project.py").exists():
    ROOT = Path.cwd().parent
elif IN_COLAB:
    print("Upload the CRM project ZIP (not a .patch).")
    uploaded = files.upload()
    zips = [name for name in uploaded if name.lower().endswith(".zip")]
    if len(zips) != 1:
        raise ValueError("Please upload exactly one project ZIP.")
    destination = Path(tempfile.mkdtemp(prefix="crm-project-"))
    with zipfile.ZipFile(zips[0]) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe ZIP path or symlink")
        archive.extractall(destination)
    candidates = list(destination.rglob("src/run_project.py"))
    if len(candidates) != 1:
        raise ValueError("ZIP must contain exactly one CRM project with src/run_project.py")
    ROOT = candidates[0].parents[1]
else:
    raise RuntimeError("Open from the project folder or set CRM_PROJECT_ROOT.")
assert (ROOT / "data/crm/sales_pipeline.csv").is_file(), "CRM data missing"
print("Project:", ROOT)
print("Host Python:", sys.version.split()[0])'''),
        nbf.v4.new_markdown_cell('''## 1. Isolated CPU environment\n
Dependencies install into the project's `.venv-crm`, not the notebook kernel. No runtime restart is needed. The skip flag below is only for an already verified local environment.'''),
        nbf.v4.new_code_cell('''if os.environ.get("CRM_NOTEBOOK_SKIP_SETUP") != "1":
    subprocess.run([sys.executable, str(ROOT / "scripts/setup_cpu.py")], cwd=ROOT, check=True)
PYTHON = Path(os.environ.get("CRM_PYTHON", str(ROOT / ".venv-crm/bin/python"))).absolute()
subprocess.run([str(PYTHON), "-m", "pip", "check"], cwd=ROOT, check=True)
print("Project interpreter:", PYTHON)'''),
        nbf.v4.new_markdown_cell('''## Regression checks\n
Run the full test suite before the reportable experiment. Tests use real small CPU fits and temporary outputs; they do not overwrite final results.'''),
        nbf.v4.new_code_cell('''subprocess.run([str(PYTHON), "-m", "pytest", "-q"], cwd=ROOT, check=True)'''),
        nbf.v4.new_markdown_cell('''## 2. Full experiment and deliverables\n
Default: full CPU budgets, all six models, SHAP, 1,589 Engaging scores, figures, paper, slides and ESL notes. `QUICK=True` is only a smoke test, never a final result. Every command stops on error.'''),
        nbf.v4.new_code_cell('''QUICK = os.environ.get("CRM_NOTEBOOK_QUICK") == "1"
OUTPUT = Path(os.environ.get("CRM_OUTPUT", str(ROOT / ("reports/crm/smoke" if QUICK else "reports/crm/final")))).resolve()
command = [str(PYTHON), "-m", "src.run_project", "--output", str(OUTPUT)]
if QUICK:
    command.append("--quick")
subprocess.run(command, cwd=ROOT, check=True)'''),
        nbf.v4.new_markdown_cell('''## 3. Verify before reporting\n
Expect status `complete`, six model rows, win+loss=1 and score counts matching the current data-quality report. The supplied snapshot has 1,589 scores and 1,088 account-missing warnings; changed valid input can have different counts. AUC near 0.5 is a warning about predictive usefulness, not a failed software run.'''),
        nbf.v4.new_code_cell('''import csv, json, math
from IPython.display import display, Markdown, Image
manifest = json.loads((OUTPUT / "run_manifest.json").read_text())
assert manifest["status"] == "complete"
with (OUTPUT / "test_metrics.csv").open() as handle:
    metrics = list(csv.DictReader(handle))
with (OUTPUT / "open_deal_predictions.csv").open() as handle:
    predictions = list(csv.DictReader(handle))
quality = json.loads((OUTPUT / "data_quality.json").read_text())
assert len(metrics) == 6
with (OUTPUT / "ablation_comparison.csv").open() as handle:
    comparison = list(csv.DictReader(handle))
assert len(comparison) == 12
agents = [json.loads(line) for line in (OUTPUT / "sales_assistant_outputs.jsonl").read_text().splitlines() if line.strip()]
assert len(agents) == len(predictions)
assert all(2 <= len(r["agent_recommendation"]["actions"]) <= 4 for r in agents)
print("PASS: 12 A/B rows and", len(agents), "assistant outputs")
for row in comparison:
    print(row["experiment"], row["model"], "test AUC", row["test_roc_auc"])
assert len(predictions) == manifest["scored_open_rows"] == quality["scorable_open_rows"]
assert all(math.isclose(float(r["win_probability"])+float(r["loss_probability"]), 1.0) for r in predictions)
assert sum(r["account_missing"] == "True" for r in predictions) == quality["scorable_missing_account"]
print("PASS:", manifest["mode"], "selected =", manifest["selected_model"])
for row in metrics:
    print(row["model"], "ROC-AUC:", round(float(row["roc_auc"]), 4), "F1:", round(float(row["f1"]), 4))
display(Image(filename=str(OUTPUT / "figures/roc_curves.png")))
display(Image(filename=str(OUTPUT / "figures/eda_missingness.png")))
display(Markdown((OUTPUT / "deliverables/RESULTS_SUMMARY.md").read_text()))'''),
        nbf.v4.new_markdown_cell('Live demo: the saved model recomputes an actual Engaging record. The rules suggest review actions, not proven interventions.'),
        nbf.v4.new_code_cell('''if manifest["scored_open_rows"]:
    subprocess.run([str(PYTHON), "-m", "src.demo", "--bundle", str(OUTPUT / "model_bundle.joblib")], cwd=ROOT, check=True)
else:
    print("No dated Engaging opportunity is available. Continue to download the results.")'''),
        nbf.v4.new_markdown_cell('''## 4. Download and keep your results\n
Colab storage is temporary. Download this ZIP before ending the session. Open the PPTX in Google Slides; inspect layout after import. Paper and slides are team-review drafts, not already submitted work.'''),
        nbf.v4.new_code_cell('''import shutil
result_zip = Path(shutil.make_archive(str(OUTPUT.parent / (OUTPUT.name + "_results")), "zip", OUTPUT))
print("Results:", result_zip)
if IN_COLAB:
    files.download(str(result_zip))''')]
    return nb


if __name__=='__main__':
    nbf.write(create_notebook(),ROOT/'notebooks/CRM_Sales_Opportunities.ipynb')
