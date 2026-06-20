"""Task auto-discovery.

On import, all task.py files under this directory are loaded and their
TASKS_TABLE entries are registered in the central registry.
"""
from mill.api.registry import load_tasks_from_path
from mill.constants import TASKS_DIR

load_tasks_from_path(str(TASKS_DIR))
