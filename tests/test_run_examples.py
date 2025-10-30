import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from heliosoil.utilities import get_project_root
import sys
import asyncio


# Fix for RuntimeWarning on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Define the root of the project to correctly locate the examples directory
project_root = get_project_root()
examples_dir = project_root / "examples"
notebook_files = list(examples_dir.glob("*.ipynb"))

# Check if notebooks are found
if not notebook_files:
    pytest.skip("No example notebooks found to test.", allow_module_level=True)


@pytest.mark.parametrize("notebook_path", notebook_files)
def test_run_notebook(notebook_path):
    """
    Executes a notebook to ensure it runs without errors.
    """
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # The executor needs to know where to find data files.
    # We set the execution path to the 'examples' directory itself.
    executor = ExecutePreprocessor(timeout=900, kernel_name="python3")

    try:
        # The second argument is a dictionary for metadata, including the path for execution.
        executor.preprocess(nb, {"metadata": {"path": examples_dir}})
    except Exception as e:
        pytest.fail(f"Error executing notebook {notebook_path.name}: {e}")
