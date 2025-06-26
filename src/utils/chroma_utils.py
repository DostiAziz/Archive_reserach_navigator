"""ChromaDB utility functions."""
import os
import chromadb
from pathlib import Path
from typing import List, Optional

from ..config import Config
from .logger_config import get_logger

logger = get_logger("chroma_utils")


def get_chroma_client():
    """
    Get a ChromaDB client based on environment.
    Uses persistent directory client for local/Docker environments,
    and HTTP client for remote/cloud deployments when configured.
    """
    # Check if we're using a remote ChromaDB service
    chroma_host = os.environ.get("CHROMA_HOST")
    chroma_port = os.environ.get("CHROMA_PORT")
    
    if chroma_host and chroma_port:
        # Using an external ChromaDB service
        logger.info(f"Connecting to ChromaDB service at {chroma_host}:{chroma_port}")
        return chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        # Using local persistence
        vector_store_path = Path(os.environ.get("VECTOR_STORE_PATH", Config.VECTOR_STORE_DIR))
        logger.info(f"Using local ChromaDB at {vector_store_path}")
        
        # Ensure directory exists
        os.makedirs(vector_store_path, exist_ok=True)
        
        return chromadb.PersistentClient(path=str(vector_store_path))


def list_chroma_collections() -> List[str]:
    """Lists the names of existing ChromaDB collections.
    
    Returns:
        List[str]: A list of collection names.
    """
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        logger.info(f"Found {len(collection_names)} existing ChromaDB collections.")
        return collection_names
    except Exception as e:
        logger.error(f"Error listing ChromaDB collections: {str(e)}")
        return []


def get_or_create_collection(name: str, embedding_function=None):
    """Gets an existing collection or creates a new one.
    
    Args:
        name: Name of the collection
        embedding_function: Optional embedding function to use
        
    Returns:
        A ChromaDB collection
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name=name, embedding_function=embedding_function)
