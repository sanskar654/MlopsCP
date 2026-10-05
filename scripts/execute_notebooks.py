"""
Notebook Executor Script
========================
Executes code cells in .ipynb files and populates outputs directly in JSON format.
"""

from __future__ import annotations

import io
import json
import sys
import trace
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def execute_notebook(notebook_path: str | Path) -> None:
    path = Path(notebook_path)
    print(f"[NOTEBOOK] Executing notebook: {path.name}...", flush=True)

    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Execution environment
    glob_env = {
        "__file__": str(path),
        "__name__": "__main__",
        "project_root": project_root,
    }

    exec_count = 1

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue

            print(f"  Running Cell {exec_count}...", flush=True)
            cell["execution_count"] = exec_count

            # Capture stdout and stderr
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            cell_outputs = []

            # Handle magic commands or display() calls
            clean_source = []
            for line in source.splitlines():
                if line.strip().startswith("%") or line.strip().startswith("!"):
                    clean_source.append(f"# {line}")
                elif line.strip().startswith("display("):
                    # Replace display(df) with print(df)
                    clean_source.append(line.replace("display(", "print("))
                else:
                    clean_source.append(line)

            code_to_exec = "\n".join(clean_source)

            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code_to_exec, glob_env)

                stdout_text = stdout_buf.getvalue()
                stderr_text = stderr_buf.getvalue()

                if stdout_text:
                    cell_outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": stdout_text.splitlines(keepends=True)
                    })

                if stderr_text:
                    cell_outputs.append({
                        "name": "stderr",
                        "output_type": "stream",
                        "text": stderr_text.splitlines(keepends=True)
                    })

            except Exception as e:
                err_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                print(f"    Cell Error: {e}")
                cell_outputs.append({
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": err_msg.splitlines(keepends=True)
                })

            cell["outputs"] = cell_outputs
            exec_count += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"[NOTEBOOK] Finished executing {path.name}! Outputs saved successfully.")


if __name__ == "__main__":
    nb1 = project_root / "notebooks" / "01_data_ingestion_and_preprocessing.ipynb"
    nb2 = project_root / "notebooks" / "02_model_training_and_evaluation.ipynb"

    execute_notebook(nb1)
    execute_notebook(nb2)
