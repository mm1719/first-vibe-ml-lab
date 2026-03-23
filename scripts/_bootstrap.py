"""Bootstrap project-root imports for script entrypoints."""

from pathlib import Path
import sys


def ensure_project_root_on_path() -> None:
    """Add the repository root to ``sys.path`` when running scripts directly."""
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
