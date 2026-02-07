"""Pytest configuration and shared fixtures."""

import shutil
from typing import Generator
import uuid
from pathlib import Path

import pytest

from src.constants import Paths


TESTS_TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


@pytest.fixture
def tmp_path_in_project() -> Generator[Path, None, None]:
    """A writable directory under tests/.tmp."""
    
    TESTS_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TESTS_TMP_ROOT / str(uuid.uuid4())
    path.mkdir(parents=True)

    yield path

    shutil.rmtree(path, ignore_errors=True)
