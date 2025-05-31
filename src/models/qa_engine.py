import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import torch
from transformers import pipeline
from transformers.utils.doc import MODELS_TO_PIPELINE

from src.models.embedding_engine import DocumentProcessor
from src.utils.logger_config import get_logger

load_dotenv()

logger = get_logger("qa_engine")


class QAEngine():
    # Constants
    DEFAULT_SEARCH_RESULTS = 50
    LOCAL_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    TEMPERATURE = 0.1

    def __init__(self, llm: str = 'genai', doc_processor: DocumentProcessor = None, ):

        if doc_processor is None:
            logger.error("No doc processor defined")
            raise ValueError("doc_processor cannot be None")

        self.emb_engine = doc_processor
        self.llm = None
        self.qa_chain = None
        self._initialize_llm(llm)
        self.emb_engine.load_vectorstore()

    def _setup_gemini_llm(self, model_name):
        """Setup google gemini llm
        Args
         :param model_name: google gemini llm name
        """
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("Gemini api key is not set")
            raise ValueError("Gemini api key is not found in the environment variables")

        self.llm = ChatGoogleGenerativeAI(model=model_name, api_key=gemini_api_key)
        logger.info(f"Google gemini llm initialized")

    def _setup_openai_llm(self, model_name: str = "gpt-3.5-turbo"):
        """Setup openai
        Args
        :param model_name: openai llm name
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("Openai api key is not set")
            raise ValueError("Openai api key is not found in the environment variables")

        self.llm = ChatOpenAI(model=model_name, api_key=openai_api_key)
        logger.info(f"OpenAI model has been initialized")

    def _setup_local_llm(self):
        """"Setup local HuggingFace LLM"""
        try:
            pipe = pipeline(
                "text-generation",
                model=self.LOCAL_MODEL_NAME,
                tokenizer=self.LOCAL_MODEL_NAME,
                do_sample=True,
                temperature=self.TEMPERATURE,
                device=0 if torch.cuda.is_available() else -1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Local LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            raise

    def _initialize_llm(self, model_id: str = 'genai'):
        """Initialize the language model
        Args:
            model_id: str the name of the language model
        """
        try:
            if model_id == 'genai':
                self._setup_gemini_llm()
            elif model_id == 'openai':
                self._setup_openai_llm()
            elif model_id == 'huggingface':
                self._setup_local_llm()
        except Exception as e:
            logger.error(f"Failed to initialize language model {model_id}: {e}")
            raise

    def perform_vectorstore_search(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve similar content from vector store
        Args:
            query (str): query to search for
            k (int): number of results to return
        """

        return self.emb_engine.query_vectorstore(query=query, k=k)

    def format_context(self, retrieved_results: List[Dict]) -> str:
        content = ""
        sources = ""
        try:
            logger.info("Formating retrieved results")
            for index, result in enumerate(retrieved_results):
                content = content + " " + result["content"]
                sources += f'{index + 1}- {result["metadata"]["title"]}\n'

            final_content = f'Content: {content} + "\n\n" + Sources:\n {sources}'
            logger.info(f'Length of formatted results {len(final_content)}')
            return final_content
        except Exception as e:
            logger.error(f"Failed to format retrieved results: {e}")
            raise

    def prompt_template(self) -> PromptTemplate:
        """Create the RAG prompt template"""
        template = """You are an AI research assistant analyzing academic papers. Use the following research paper
               excerpts to answer the question. Be accurate, cite specific papers when possible, and acknowledge when
               information is not available.

               Research Paper Context:
               {context}
               Question: {question}
               Instructions:
               1. Provide a comprehensive answer based on the research papers.
               2. Mention specific papers or authors when relevant, using the source numbers (e.g., [1], [2]).
               3. If the papers don't contain enough information, say so clearly.
               4. Use academic language appropriate for research.
               5. At the end of your generated answer, list the sources you have used as a reference, in this format:
               References:
               [1] Title of the paper goes here.
               [2] Title of another paper.
               Only cite papers that are directly relevant to the answer.
               Answer:
               """
        return PromptTemplate(
            template=template,
            input_variables=['context', 'question']
        )

    def generate_answer(self, query: str) -> str:
        """Generate an answer using RAG approach
        Args:
            query (str): query to search for
        Returns:
            :param query: The question for llm to answer
        """
        try:
            # Get context from vector search
            retrieved_results = self.perform_vectorstore_search(query=query, k=self.DEFAULT_SEARCH_RESULTS)
            context = self.format_context(retrieved_results)

            if not context:
                logger.warning(f"Failed to format retrieved results: {query}")
                context = "No relevant research papers found for this query"

            prompt = self.prompt_template()
            # Format the prompt with actual values
            formatted_prompt = prompt.format(
                context=context,
                question=query
            )

            # Invoke LLM with formatted prompt
            response = self.llm.invoke(formatted_prompt)

            # Return content based on a response type
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error while processing your query: {str(e)}"


if __name__ == '__main__':
    from src.models.embedding_engine import DocumentProcessor

    qa_engine = QAEngine(llm='huggingface', doc_processor=DocumentProcessor())
