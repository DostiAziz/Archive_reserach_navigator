import pandas as pd
import torch
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

from langchain_core.documents import Document

from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model,
                                                     model_kwargs={'device': device},
                                                     encode_kwargs={'normalize_embeddings': True}
                                                     )
        self.persistent_directory = Config.VECTOR_STORE_DIR
        os.makedirs(self.persistent_directory, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE,
                                                            chunk_overlap=Config.OVERLAPP,
                                                            length_function=len)

        self.vectorstore = None
        logger.info(f"Document processor initialized")

    def prepare_documents(self, paper_df: pd.DataFrame) -> List[Document]:
        """Convert dataframe of retried paper information to proper langchain documents.
        Args:
            paper_df (pd.DataFrame): paper data frame
        Returns:
            List[Document]: list of langchain documents
        """

        logger.info(f'converting {paper_df.shape[0]} paper documents')
        documents = []
        try:
            for index, paper in paper_df.iterrows():
                content = f"Title: {paper.title}\n\nAbstract: {paper.abstract}\n\n"

                metadata = {
                    'title': paper.title,
                    'doi': paper.doi,
                    'authors': paper.authors,
                    'categories': paper.categories,
                    'published': paper.published,
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            logger.info(f'converted {len(documents)} langchain documents')
            return documents
        except Exception as e:
            logger.error(f'failed to convert {paper_df.shape[0]} paper documents')
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
        """Create ChromaDB vectorstore from chunked documents
        Args:
            documents (List[Document]): list of chunked documents
            collection_name (str, optional): collection name. Defaults to 'research_papers'.
        Returns:
            None

        """
        try:
            logger.info(f'building {collection_name} vectorstore')
            self.vectorstore = Chroma(collection_name=collection_name, embedding_function=self.embedding_model,
                                      persist_directory=self.persistent_directory)
            logger.info('Adding documents to vectorstore')
            self.vectorstore.add_documents(documents)

            logger.info(f' Collection {collection_name} contains {self.vectorstore._collection.count()} documents')

        except Exception as e:
            logger.error(f'failed to build {collection_name} vectorstore')
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

    def query_vectorstore(self, query: str, k: int = 5, score_threshold: float = 0.8) -> List[Dict]:
        """Query the knowledge base using sematic similarity
        Args:
            query (str): question to be asked
            k (int, optional): number of similar documents to return. Defaults to 5.
            score_threshold (float, optional): threshold for similarity. Defaults to 0.8.
        Returns:
            List[Dict]: list of dictionaries containing relevant information

        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k, score_threshold=score_threshold)
            formatted_results = []
            for index, (doc, score) in enumerate(results):
                if score > score_threshold:
                    result = {'rank': index + 1,
                              'similarity_score': score,
                              'content': doc.page_content,
                              'metadata': doc.metadata
                              }
                    formatted_results.append(result)
            logger.info(f'Query {query} returned {len(results)} relevant papers')
            return formatted_results
        except Exception as e:
            logger.error(f'failed to query {query}')
            raise e

    def get_retriever(self, search_kwargs: Dict = None):
        """Get langchain retriever for RAG
        Args:
            search_kwargs (Dict): parameter for search (e.g {'query':AI, 'k':5}
        """

        try:
            if not self.vectorstore:
                logger.info(f'retrieving vectorstore from {self.vectorstore._collection.count()} documents')
                raise ValueError('no vectorstore found, create or initialize vectorstore first')

            if search_kwargs is None:
                search_kwargs = {'k': 5}

            return self.vectorstore.get_retriever(**search_kwargs)
        except Exception as e:
            logger.error(f'failed to retrieve retriever for {search_kwargs}')
            raise e


if __name__ == '__main__':
    from data_pipeline import DataPipeline

    pipeline = DataPipeline()
    results = pipeline.search_paper(query="RAG", max_results=200)
    results_df = pd.DataFrame(results)
    doc_processor = DocumentProcessor()
    prepared_doc = doc_processor.prepare_documents(results_df)
    chunked_docs = doc_processor.chunk_documents(prepared_doc)

    #create vector store
    doc_processor.build_vectorstore(chunked_docs)


    queries = [
            "Deep learning neural networks",
            "natural language processing",
            "rag for medicine"
    ]


    print(f'\n'+ "="*50)

