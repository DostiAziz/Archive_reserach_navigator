import os
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging

import torch
from transformers import pipeline

from src.models.embedding_engine import DocumentProcessor

from src.models.config import Config
from src.models.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


class QAEngine():
    def __init__(self, llm: str = 'genai', doc_processor: DocumentProcessor = None, ):

        self.emb_engine = doc_processor
        self.llm = None
        self.qa_chain = None

        self._initialize_llm(llm)
        self.emb_engine.load_vectorstore()

    def _initialize_llm(self, model_id: str = 'genai'):

        if model_id == 'genai' and os.environ.get("GEMINI_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=Config.GEMINI_API_KEY)

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

    def format_context(self, retrieved_results: List[Dict]) -> str:
        content = ""
        sources = ""
        try:
            for index, result in enumerate(retrieved_results):
                content = content + " " + result["content"]
                sources += f'{index + 1}- {result["metadata"]["title"]}\n'
            final_content = f'Content: {content} + "\n\n" + Sources:\n {sources}'
            logger.info(f'Formatted results {final_content}')
            return final_content
        except Exception as e:
            logger.error(f"Failed to format retrieved results: {e}")
            raise

    def generate_answer(self, query: str) -> str:
        """Generate answer using RAG approach"""
        try:
            # Get context from vector search
            context = self.format_context(
                self.perform_vectorstore_search(query=query, k=50)
            )

            template = """You are an AI research assistant analyzing academic papers. Use the following research paper 
            excerpts to answer the question. Be accurate, cite specific papers when possible, and acknowledge when 
            information is not available. 
            

            Research Paper Context:
            {context}

            Question: {question}

            Instructions:
            1. Provide a comprehensive answer based on the research papers
            2. Mention specific papers or authors when relevant
            3. If the papers don't contain enough information, say so clearly
            4. Use academic language appropriate for research
            
            When citing paper inside the paragraph only use number that is in front of that paper. 
            At the end of the your generated answer, list the sources you have used as a reference, in this format.
            
            1- Title of the paper goes here. 

            May be the number of returned paper will be large only cite these papers which seems really relevant to the query.
            Answer:
            """

            # Create prompt template
            prompt = PromptTemplate(
                template=template,  # Your template string
                input_variables=['context', 'question']
            )

            # Format the prompt with actual values
            formatted_prompt = prompt.format(
                context=context,
                question=query
            )

            # Invoke LLM with formatted prompt
            response = self.llm.invoke(formatted_prompt)

            # Return content based on response type
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error: {str(e)}"


if __name__ == '__main__':
    from src.models.embedding_engine import DocumentProcessor

    qa_engine = QAEngine(doc_processor=DocumentProcessor())
