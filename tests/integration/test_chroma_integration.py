"""Integration tests for ChromaDB utilities."""
import os
import pytest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.chroma_utils import get_chroma_client, list_chroma_collections, get_or_create_collection


class TestChromaDBIntegration:
    """Integration tests for ChromaDB functions."""

    @pytest.mark.integration
    def test_create_and_list_collections(self, temp_dir):
        """Test creating and listing collections using a real ChromaDB instance."""
        # Set up a test environment with a temporary directory
        os.environ["VECTOR_STORE_PATH"] = str(temp_dir)
        
        # Create test collections
        client = get_chroma_client()
        collection_names = ["test_collection_1", "test_collection_2"]
        
        for name in collection_names:
            client.create_collection(name=name)
        
        # Get the collections using our function
        result = list_chroma_collections()
        
        # Verify the collections were created and listed
        assert len(result) == 2
        assert set(result) == set(collection_names)

    @pytest.mark.integration
    def test_get_or_create_collection(self, temp_dir):
        """Test get_or_create_collection function with a real ChromaDB instance."""
        # Set up a test environment with a temporary directory
        os.environ["VECTOR_STORE_PATH"] = str(temp_dir)
        
        # Create a collection
        collection_name = "test_get_or_create"
        collection = get_or_create_collection(collection_name)
        
        # Verify the collection was created
        assert collection.name == collection_name
        
        # Try getting the same collection again
        collection2 = get_or_create_collection(collection_name)
        
        # Verify we got the same collection
        assert collection2.name == collection_name
        
        # Check that it's in the list of collections
        collections = list_chroma_collections()
        assert collection_name in collections
