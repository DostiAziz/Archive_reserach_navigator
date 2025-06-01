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
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model,
                                                     model_kwargs={'device': device},
                                                     encode_kwargs={'normalize_embeddings': True}
                                                     )
        self.persistent_directory = Config.VECTOR_STORE_DIR
        os.makedirs(self.persistent_directory, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE,
                                                            chunk_overlap=Config.OVERLAP,
                                                            length_function=len)

        self.vectorstore = None
        logger.info(f"Document processor initialized")

    def prepare_documents(self, paper_df: pd.DataFrame) -> List[Document]:
        """Convert dataframe containng retrieved paper information for langchain documents
        Args:
            paper_df: paper dataframe
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
                content = f"Title: {title}\nAbstract: {abstract}"

                # create metadata for each document
                metadata = {
                    'title': title,
                    'id': paper.id,
                    'authors': paper.authors,
                    'categories': paper.categories,
                    'published': paper.published,
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
                # retrieve database instance and delete it if exists
                client = chromadb.PersistentClient(path=self.persistent_directory)
                client.delete_collection(name=collection_name)
                logger.info(f'Successfully deleted {collection_name} vectorstore')
            except Exception as e:
                logger.error(f'failed to create {collection_name} vectorstore: {e}')

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

    def query_vectorstore(self, query: str, k: int = 5) -> List[Dict]:
        """Query the knowledge base using sematic similarity
        Args:
            query (str): question to be asked
            k (int, optional): number of similar documents to return. Defaults to 5.
        Returns:
            List[Dict]: list of dictionaries containing relevant information
        """
        try:
            logger.info(f'Querying knowledge base using {query}')
            # get more results to allow for better filtering
            query_results = self.vectorstore.similarity_search_with_score(query, k=k)

            # sort returned results from most similar to less
            query_results = sorted(query_results, key=lambda x: x[1], reverse=True)
            formatted_results = []

            for index, (doc, score) in enumerate(query_results):
                result = {'rank': index + 1,
                          'similarity_score': score,
                          'content': doc.page_content,
                          'metadata': doc.metadata
                          }
                formatted_results.append(result)

            logger.info(f'Query {query} returned {len(query_results)} relevant papers')
            return formatted_results

        except Exception as e:
            logger.error(f'failed to query {query}')
            raise e

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
