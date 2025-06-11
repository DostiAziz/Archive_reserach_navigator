import os
from typing import List, Dict
import pandas as pd
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from utils.logger_config import get_logger

from config import Config

logger = get_logger("embedding-engine")


class DocumentProcessor:

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Embedding model for creating vector database
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model,
                                                     model_kwargs={'device': device},
                                                     encode_kwargs={'normalize_embeddings': True}
                                                     )
        self.persistent_directory = Config.VECTOR_STORE_DIR
        os.makedirs(self.persistent_directory, exist_ok=True)

        # Used for chunking of retreived documents
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE,
                                                            chunk_overlap=Config.OVERLAP,
                                                            length_function=len)

        self.vectorstore = None
        self.current_author_names = []

        logger.info(f"Document processor initialized")

    def prepare_documents(self, paper_df: pd.DataFrame) -> List[Document]:
        """Convert dataframe containng retrieved paper information for langchain documents
        Args:
            paper_df (pd.DataFrame): papers dataframe
        Returns:
            List[Document] list containing paper documents in langchain format
        """
        logger.info(f'converting {paper_df.shape[0]} paper documents')
        documents = []
        try:
            for index, paper in tqdm(paper_df.iterrows(), desc="Creating langchain documents", total=paper_df.shape[0]):
                # create content by combining title and abstract
                title = paper.title
                abstract = paper.abstract
                authors = paper.authors
                published = paper.published.split('T')[0] if paper.published else None

                content = f"Title: {title}\n Abstract: {abstract} \n Authors: {authors} \n Publication year: {published}"

                # create metadata for each document
                metadata = {
                    'title': title,
                    'id': paper.id,
                    'authors': authors,
                    'categories': paper.categories,
                    'year-published': published,

                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f'failed to convert {paper_df.shape[0]} paper documents: {e}')
            raise e

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split each document into chunk with specified size and overlap
        Args:
            documents (List[Document]): list of documents
        Returns:
            List[Document]: list of chunks of documents
       """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info('finished chunking documents')
            return chunks
        except Exception as e:
            logger.error(f'failed to split documents: {e}')
            raise e

    def build_vectorstore(self, documents: List[Document], collection_name: str = 'research_papers'):
        """Builds verctor database for the list of langchain documents,
        first deletes exising vector databases for that collection name
        Args:
            documents (List[Document]): list of documents
            collection_name (str, optional): collection name. Defaults to 'research_papers'.
        Returns:
            None
        """
        try:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.persistent_directory)
                collection_names = [col.name for col in client.list_collections()]
                logger.info(f'Existing collections: {collection_names}')

                if collection_name in collection_names:
                    logger.info(f'Found existing collection {collection_name}, attempting to delete')
                    client.delete_collection(name=collection_name)
                    logger.info(f'Successfully deleted old {collection_name} vectorstore')
            except Exception as e:
                logger.error(f'failed to delete old instances of {collection_name} vectorstore: {e}')

            logger.info(f'Building {collection_name} vectorstore')

            # Create an empty vectorstore
            self.vectorstore = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.persistent_directory,
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"}
            )

            # Add documents in batches
            batch_size = 100
            total_added = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
                total_added += len(batch)
                logger.info(f'Added batch {i // batch_size + 1}, total documents: {total_added}')

            logger.info(f'Collection {collection_name} contains {self.vectorstore._collection.count()} documents')

        except Exception as e:
            logger.error(f'failed to build {collection_name} vectorstore: {e}')
            raise e

    def load_vectorstore(self, collection_name: str = 'research_papers'):

        """Load vectorstore from collection
        Args:
            collection_name (str, optional): collection name. Defaults to 'research_papers'.
        """
        try:
            self.vectorstore = Chroma(collection_name=collection_name, embedding_function=self.embedding_model,
                                      persist_directory=self.persistent_directory)
            logger.info(f'Vectorstore loaded with {self.vectorstore._collection.count()} documents')
        except Exception as e:
            logger.error(f'failed to load {collection_name} vectorstore')

    def query_vectorstore(self, query: str, is_author_query: bool, author_names: List[str], k: int = 5) -> List[Dict]:
        """Query the knowledge base using semantic similarity and metadata filtering
        Args:
            query (str): search query
            is_author_query (bool): whether the query is related to authors
            author_names (List[str]): list of author names to filter by
            k (int): number of results to return

        """
        try:
            logger.info(f'Querying knowledge base using {query}')

            if is_author_query and author_names:
                logger.info(f'Author query detected, filtering by author: {author_names}')
                # Store author names for manual filtering
                self.current_author_names = author_names
                logger.info("Using manual filtering for partial author name matching")
                query_results = self._manual_author_filtering(query, k)
            else:
                logger.info("Query is not realted to authors, so perform standard similarity search")
                self.current_author_names = []
                query_results = self.vectorstore.similarity_search_with_score(query=query, k=k)

            formatted_results = []
            query_results = sorted(query_results, key=lambda x: x[1], reverse=True)
            for index, (doc, score) in enumerate(query_results):
                result = {
                    'rank': index + 1,
                    'similarity_score': score,
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                formatted_results.append(result)

            logger.info(f'Query "{query}" returned {len(formatted_results)} relevant papers')
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying vectorstore: {e}")
            return []

    def _manual_author_filtering(self, query: str, k: int):
        """Manual filtering for partial author name matching
        Args:
            query (str): search query
            k (int): number of results to return
        """
        try:
            logger.info(f"Manual filtering for authors: {self.current_author_names}")

            # Get more results to filter from (increase multiplier for better results)
            search_k = min(k * 20, 200)  # Get up to 200 docs to filter from
            all_results = self.vectorstore.similarity_search_with_score(query=query, k=search_k)

            logger.info(f"Retrieved {len(all_results)} documents to filter")

            filtered_results = []

            for doc, score in all_results:
                # Convert authors field to string for matching
                authors_field = str(doc.metadata.get('authors', ''))

                # Check if any author name appears in the authors field (partial matching)
                match_found = False
                for author_name in self.current_author_names:
                    if self._is_author_match(author_name, authors_field):
                        match_found = True
                        logger.info(f"✓ Found match for '{author_name}' in: {authors_field}")
                        break

                if match_found:
                    filtered_results.append((doc, score))

                # if len(filtered_results) >= k:
                #     break

            logger.info(f"Manual filtering found {len(filtered_results)} results out of {len(all_results)} total")

            # If we didn't find enough results, log some examples of what we did find
            if len(filtered_results) < k and len(all_results) > 0:
                logger.info("Sample authors fields from unmatched documents:")
                for i, (doc, _) in enumerate(all_results[:5]):
                    sample_authors = doc.metadata.get('authors', 'No authors')
                    logger.info(f"  Sample {i + 1}: {sample_authors}")

            return filtered_results

        except Exception as e:
            logger.error(f"Manual filtering error: {e}")
            return []

    def _is_author_match(self, search_author: str, authors_field: str) -> bool:
        """Check if search author matches any author in the authors field
        Args:
            search_author (str): author name to search for
            authors_field (str): authors field from the document metadata
        Returns:
            bool: True if a match is found, False otherwise
        """
        # Check if search_author or authors_field is empty
        if not search_author or not authors_field:
            return False

        # Convert to lowercase for case-insensitive matching
        search_lower = search_author.lower().strip()
        authors_lower = authors_field.lower()

        # Split by comma and check each author
        authors_list = [author.strip() for author in authors_lower.split(',') if author.strip()]

        for author in authors_list:
            if search_lower in author or author in search_lower:
                return True

        return False

    def get_retriever(self, search_kwargs: Dict = None):
        """Get langchain retriever for RAG"""
        try:
            if not self.vectorstore:
                raise ValueError('No vectorstore found, create or initialize vectorstore first')

            logger.info(f'Creating retriever from vectorstore with {self.vectorstore._collection.count()} documents')

            if search_kwargs is None:
                search_kwargs = {'k': 5}

            return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        except Exception as e:
            logger.error(f'Failed to create retriever: {e}')
            raise e

#
# if __name__ == '__main__':
#     from models.data_pipeline import DataPipeline
#     from models.config import Config
#
#     pipeline = DataPipeline()
#     results = pipeline.search_paper(query="RAG", max_results=2000)
#     results_df = pd.DataFrame(results)
#
#     doc_processor = DocumentProcessor()
#     prepared_doc = doc_processor.prepare_documents(results_df)
#     # uncomment this line if you want to chunk the files
#     # prepared_doc = doc_processor.chunk_documents(prepared_doc)
#
#     # Create a vector store
#     doc_processor.build_vectorstore(prepared_doc)
