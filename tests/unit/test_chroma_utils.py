"""Unit tests for ChromaDB utilities."""
import os
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.chroma_utils import list_chroma_collections, get_chroma_client


class TestChromaUtils:
    """Tests for ChromaDB utilities."""

    @patch('src.utils.chroma_utils.get_chroma_client')
    def test_list_chroma_collections_success(self, mock_get_client):
        """Test listing ChromaDB collections when successful."""
        # Set up the mock
        mock_client = MagicMock()
        mock_collection1 = MagicMock()
        mock_collection2 = MagicMock()
        mock_collection1.name = "test_collection_1"
        mock_collection2.name = "test_collection_2"
        mock_client.list_collections.return_value = [mock_collection1, mock_collection2]
        mock_get_client.return_value = mock_client
        
        # Execute the function
        result = list_chroma_collections()
        
        # Assertions
        assert len(result) == 2
        assert "test_collection_1" in result
        assert "test_collection_2" in result
        mock_client.list_collections.assert_called_once()

    @patch('src.utils.chroma_utils.get_chroma_client')
    def test_list_chroma_collections_empty(self, mock_get_client):
        """Test listing ChromaDB collections when none exist."""
        # Set up the mock
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_get_client.return_value = mock_client
        
        # Execute the function
        result = list_chroma_collections()
        
        # Assertions
        assert len(result) == 0
        assert isinstance(result, list)
        mock_client.list_collections.assert_called_once()

    @patch('src.utils.chroma_utils.get_chroma_client')
    def test_list_chroma_collections_error(self, mock_get_client):
        """Test listing ChromaDB collections when an error occurs."""
        # Set up the mock to raise an exception
        mock_get_client.side_effect = Exception("Connection failed")
        
        # Execute the function
        result = list_chroma_collections()
        
        # It should handle the exception and return an empty list
        assert len(result) == 0
        assert isinstance(result, list)

    @patch('chromadb.PersistentClient')
    @patch('chromadb.HttpClient')
    @patch('os.environ.get')
    def test_get_chroma_client_local(self, mock_environ_get, mock_http_client, mock_persistent_client):
        """Test getting a ChromaDB client for local usage."""
        # Set up the mocks
        mock_environ_get.side_effect = lambda key, default=None: None
        mock_persistent_client.return_value = MagicMock()
        
        # Execute the function
        client = get_chroma_client()
        
        # Assertions
        assert client is not None
        mock_persistent_client.assert_called_once()
        mock_http_client.assert_not_called()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.HttpClient')
    @patch('os.environ.get')
    def test_get_chroma_client_remote(self, mock_environ_get, mock_http_client, mock_persistent_client):
        """Test getting a ChromaDB client for remote service."""
        # Set up the mocks
        mock_environ_get.side_effect = lambda key, default=None: "localhost" if key == "CHROMA_HOST" else "8000" if key == "CHROMA_PORT" else None
        mock_http_client.return_value = MagicMock()
        
        # Execute the function
        client = get_chroma_client()
        
        # Assertions
        assert client is not None
        mock_http_client.assert_called_once_with(host="localhost", port="8000")
        mock_persistent_client.assert_not_called()
