import os 
from pathlib import Path
from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _resolve_project_path(filepath:str)->str:
    """Converts a filepath into an absolute path inside the project."""
    path=(PROJECT_ROOT/ filepath).resolve()

    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Invalid filepath.")
    return path

@tool
def create_file(filepath:str)->str:
    """Create an empty file under the project root."""
    path=_resolve_project_path(filepath)
    path.parent.mkdir(parent=True,exist_ok=True)
    path.touch(exist_ok=True)
    return f"Created file at {path.relative_to(PROJECT_ROOT)}"

@tool
def write_file(filepath:str,content:str)->str:
    """Write content to a file under project root."""
    path = _resolve_project_path(filepath)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding="utf-8")

    return f"Write content to {path.relative_to(PROJECT_ROOT)}"

