"""
Tests for shared/config.py
"""

import pytest
from shared.config import (
    PROJECT_ROOT,
    DATA_DIR,
    VECTOR_STORE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_K,
)


def test_project_root_exists():
    """Test PROJECT_ROOT is valid"""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_data_dir_path():
    """Test DATA_DIR path construction"""
    assert "data" in str(DATA_DIR)
    assert DATA_DIR.parent == PROJECT_ROOT


def test_vector_store_dir():
    """Test VECTOR_STORE_DIR path"""
    assert "vector_stores" in str(VECTOR_STORE_DIR)


def test_default_constants():
    """Test default configuration constants"""
    assert DEFAULT_CHUNK_SIZE == 1000
    assert DEFAULT_K == 4
    assert isinstance(DEFAULT_CHUNK_SIZE, int)
    assert isinstance(DEFAULT_K, int)
