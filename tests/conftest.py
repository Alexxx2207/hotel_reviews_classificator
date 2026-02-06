"""Pytest configuration and shared fixtures."""

import shutil
import uuid
from pathlib import Path

import pytest

from src.constants import Paths


# Directory under tests/ for temp data (avoids system temp / PermissionError on Windows)
TESTS_TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


@pytest.fixture
def tmp_path_in_project() -> Path:
    """A unique writable directory under tests/.tmp (no system temp)."""
    
    TESTS_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TESTS_TMP_ROOT / str(uuid.uuid4())
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def tmp_project(tmp_path_in_project: Path) -> Paths:
    """Paths instance with project_root set to a temp dir under tests/."""

    return Paths(project_root=tmp_path_in_project)
