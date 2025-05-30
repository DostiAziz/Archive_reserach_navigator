from typing import List, Dict

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import con
from langchain.prompts import Prompt, PromptTemplate
import logging

from sympy.printing.pytorch import torch
from transformers import pipeline

from data_pipeline import DataPipeline
from embedding_engine import DocumentProcessor
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)


class QAEngine():
    def __init__(self, llm: str = 'genai'):

        self.emb_engine = DocumentProcessor()
        self.llm = None
        self.qa_chain = None

        self._initialize_llm(llm)

    def _initialize_llm(self, model_id: str = 'genai'):

        if model_id == 'genai':
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        elif model_id == 'huggingface':
            model_name = "microsoft/DialoGPT-medium"

            # Create HuggingFace pipeline
            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                device=0 if torch.cuda.is_available() else -1
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("✅ Local LLM loaded successfully")
        elif model_id == 'openai':
            self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    def perform_vectorstore_search(self, query: str, k: int = 5):
        """Retrive similar content from vector store
        Args:
            query (str): query to search for
            k (int): number of results to return
        """

        return self.emb_engine.query_vectorstore(query=query, k=k)

    def format_contxt(self, retrieved_results: List[Dict]) -> str:

        content = ''.join(content for content in retrieved_results['page_content'])
        sources = '\n'.join(title for title in retrieved_results['metadata']['titles'])
        return f'content: {content}\nsources: {sources}'

    def generate_answer(self, query: str) -> str:

        if self.llm is None:
            logger.error("language model is not initialized")
            raise

        template = """You are an AI research assistant analyzing academic papers. Use the following research paper 
        excerpts to answer the question. Be accurate, cite specific papers when possible, and acknowledge when 
        information is not available. At the end of research paper context there is list of of sources which these information
        if collected, remember to cite these sources at the end of your response.

        Research Paper Context:
        {context}

        Question: {question}

        Instructions:
        1. Provide a comprehensive answer based on the research papers
        2. Mention specific papers or authors when relevant
        3. If the papers don't contain enough information, say so clearly
        4. Use academic language appropriate for research

        Answer:
        """
        context = self.format_contxt(self.perform_vectorstore_search(query=query, k=10))

        chat_memory = ConversationBufferMemory()
        prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question'])
        prompt.format(context=context, question=query)




