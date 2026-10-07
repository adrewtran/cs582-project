import json
import os
from pathlib import Path
import subprocess
import sys
import nbformat


def test_notebook_setup_keeps_the_virtualenv_interpreter(tmp_path):
    root=Path(__file__).resolve().parents[1]
    notebook=nbformat.read(root/'notebooks/CRM_Sales_Opportunities.ipynb',as_version=4)
    code=[c.source for c in notebook.cells if c.cell_type=='code']
    namespace={}
    old={k:os.environ.get(k) for k in ['CRM_PROJECT_ROOT','CRM_NOTEBOOK_SKIP_SETUP','CRM_PYTHON']}
    os.environ.update(CRM_PROJECT_ROOT=str(root),CRM_NOTEBOOK_SKIP_SETUP='1',CRM_PYTHON=sys.executable)
    try:
        exec(code[0],namespace)
        exec(code[1],namespace)
        actual=subprocess.check_output([str(namespace['PYTHON']),'-c','import sys; print(sys.prefix)'],text=True).strip()
        assert actual==sys.prefix
    finally:
        for k,value in old.items():
            if value is None: os.environ.pop(k,None)
            else: os.environ[k]=value
