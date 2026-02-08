"""Pytest configuration and shared fixtures."""

import shutil
from typing import Generator
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path_in_project() -> Generator[Path, None, None]:
    """A writable directory under tests/.tmp."""

    tmp_root = Path(__file__).resolve().parent / ".tmp"

    tmp_root.mkdir(parents=True, exist_ok=True)
    path = tmp_root / str(uuid.uuid4())
    path.mkdir(parents=True)

    yield path

    shutil.rmtree(path, ignore_errors=True)
