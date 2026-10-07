from pathlib import Path
import ast
import nbformat


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "CRM_Sales_Opportunities.ipynb"


def test_notebook_is_valid_and_every_cell_is_executable_python():
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    nbformat.validate(notebook)
    code=[cell.source for cell in notebook.cells if cell.cell_type=='code']
    assert len(code)>=4
    for source in code: ast.parse(source)
