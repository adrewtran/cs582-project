"""Execute every notebook cell locally; records evidence, not a Colab claim."""
import json
import os
from pathlib import Path
import sys
import tempfile
import contextlib
import io
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def main():
    evidence=ROOT/'reports/crm/final/verification'; evidence.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='crm-notebook-') as temp:
        os.environ.update(CRM_PROJECT_ROOT=str(ROOT),CRM_NOTEBOOK_SKIP_SETUP='1',
            CRM_PYTHON=sys.executable,CRM_OUTPUT=str(Path(temp)/'notebook_run'),CRM_NOTEBOOK_QUICK='0')
        notebook=nbformat.read(ROOT/'notebooks/CRM_Sales_Opportunities.ipynb',as_version=4)
        # This workspace prohibits both TCP and IPC listeners, so a Jupyter
        # kernel cannot start. Execute the unchanged Python cells sequentially
        # instead, and label this narrower verification explicitly.
        namespace={}; count=0
        for cell in notebook.cells:
            if cell.cell_type!='code': continue
            count+=1; captured=io.StringIO()
            with contextlib.redirect_stdout(captured):
                exec(compile(cell.source,f'notebook-cell-{count}','exec'),namespace)
            cell.execution_count=count
            cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=captured.getvalue())]
        nbformat.write(notebook,evidence/'CRM_Notebook_Executed.ipynb')
        manifest=json.loads((Path(temp)/'notebook_run/run_manifest.json').read_text())
        code_cells=[cell for cell in notebook.cells if cell.cell_type=='code']
        report={'status':'passed','execution':'unchanged notebook code cells in one Python process; not a Jupyter/Colab kernel',
                'code_cells_executed':len(code_cells),'mode':manifest['mode'],
                'pipeline_status':manifest['status'],'scored_open_rows':manifest['scored_open_rows'],
                'setup':'already-installed isolated runtime; setup_cpu.py tested separately',
                'not_verified':['Jupyter kernel: local socket binding prohibited','Google account login','Colab upload/download UI','Google Slides import']}
        (evidence/'notebook_execution.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__': main()
