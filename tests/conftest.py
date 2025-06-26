"""Pytest configuration file."""
import os
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_environment():
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()
    os.environ["VECTOR_STORE_PATH"] = "/tmp/test_vector_store"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["ENVIRONMENT"] = "test"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def chroma_test_collections():
    """Return a list of test collection names."""
    return ["test_collection_1", "test_collection_2", "test_collection_3"]
