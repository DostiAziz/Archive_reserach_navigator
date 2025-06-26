import os
from typing import List, Dict
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import pipeline

from models.embedding_engine import DocumentProcessor
from utils.logger_config import get_logger

load_dotenv()

logger = get_logger("qa_engine")


class QAEngine():
    """Class for QA engine """

    # Constants
    DEFAULT_SEARCH_RESULTS = 5
    LOCAL_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    TEMPERATURE = 0.1

    def __init__(self, llm: str = 'genai', doc_processor: DocumentProcessor = None, vs_instance_name: str = ""):

        if doc_processor is None:
            logger.error("No doc processor defined")
            raise ValueError("doc_processor cannot be None")

        self.emb_engine = doc_processor
        self.llm = None
        self.qa_chain = None

        # Initialize llm
        self._initialize_llm(llm)
        # Load vector database
        self.emb_engine.load_vectorstore(collection_name=vs_instance_name)

    def _setup_gemini_llm(self, model_name='gemini-2.0-flash'):
        """Setup google gemini llm
        Args
         model_name (str): model name default gemini-2.0-flash
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
            model_name (str): Defaults to "gpt-3.5-turbo".
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
            )
            self.llm = HuggingFacePipeline(pipeline=pipe,
                                           pipeline_kwargs={"return_full_text": False}
                                           )
            # Skip printing the prompt
            self.llm.bind(skip_prompt=True)
            logger.info("Local LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            raise

    def _initialize_llm(self, model_id: str = 'genai'):
        """Initialize the language model
        Args:
            model_id: str the name of the language model
        Returns:
            None
        """
        try:
            if model_id == 'genai':
                self._setup_gemini_llm(model_name="gemini-2.0-flash")
            elif model_id == 'openai':
                self._setup_openai_llm()
            elif model_id == 'huggingface':
                self._setup_local_llm()
        except Exception as e:
            logger.error(f"Failed to initialize language model {model_id}: {e}")
            raise

    def perform_vectorstore_search(self, query: str, is_author_query: bool, author_names: List[str], k: int = 5) -> \
            List[
                Dict]:
        """Retrieve similar content from vector store
        Args:
            query (str): query to search for in vector store
            k (int): number of results to return
            is_author_query (bool): whether the query is about authors
            author_names (List[str]): list of author names to search for
        """

        return self.emb_engine.query_vectorstore(query=query, is_author_query=is_author_query,
                                                 author_names=author_names,
                                                 k=k)

    def format_context(self, retrieved_results: List[Dict]) -> str:
        """Format the retrieved results to separate context from the references
        Args:
            retrieved_results (List[Dict]): list of retrieved results
        Returns:
            str: formatted context
        """

        content = ""
        sources = ""
        try:
            logger.info("Formatting retrieved results")

            # Create a structured format that makes author information prominent
            content += "=== RESEARCH PAPERS DATABASE ===\n\n"

            for index, result in enumerate(retrieved_results):
                doc_num = index + 1
                metadata = result.get("metadata", {})

                # Extract metadata with fallbacks
                title = metadata.get("title", "Unknown Title")
                authors = metadata.get("authors", "Unknown Authors")
                abstract = result.get("content", "Unknown Abstract")
                year_published = metadata.get("year-published", "Unknown Year")

                # Format each paper clearly
                content += f"[PAPER {doc_num}]\n"
                content += f"TITLE: {title}\n"
                content += f"AUTHORS: {authors}\n"
                content += f"ABSTRACT: {abstract}\n"
                content += f"PUBLISHED YEAR: {year_published}\n"
                content += "=" * 80 + "\n\n"

                # Build sources reference list
                sources += f'[{doc_num}] {title}\n'
                sources += f" Authors: {authors}\n\n"

            final_content = f'{content}\nSOURCES SUMMARY:\n{sources}'
            return final_content
        except Exception as e:
            logger.error(f"Failed to format retrieved results: {e}")
            raise

    def prompt_template(self) -> PromptTemplate:
        """Create the RAG prompt template
        Returns:
            PromptTemplate: RAG prompt template
        """
        template = """You are an AI research assistant analyzing academic papers. Use the following research paper
               excerpts to answer the question. Understand what the user is asking. Each paper includes Abstract and metadata with authors, titles, publication year.
               Be accurate, cite specific papers when possible, and acknowledge when information is not available.

               Research Paper Context:
               {context}
               Question: {question}

               Instructions:
               1. Pay special attention to author names, the author names are included in the context.
               2. When asked about authors, search through ALL the paper and try to find where author names are mentioned.
               4. Mention specific papers or authors when relevant, using the source numbers (e.g., [1], [2]).
               5. If you cannot find information about a specific author, clearly state "No papers by [author name] were found in the provided context."
               6. Use academic language appropriate for research.
               7. At the end of your generated answer, list the sources you have used as a reference, in this format:

              References:

                [1] Title of first paper - Authors

                [2] Title of second paper - Authors

                [3] Title of third paper - Authors

                Only cite papers that are directly relevant to the answer. 

                Make sure each reference is on its own line with proper spacing.

               """
        return PromptTemplate(
            template=template,
            input_variables=['context', 'question']
        )

    def author_detector_prompt_simple(self) -> PromptTemplate:
        """Create a simple author detector prompt template
        Returns:
            PromptTemplate: author detector prompt template
        """
        template = """You are an AI assistant that detects if a query is asking about specific authors.

        Query: "{query}"

        Answer with EXACTLY this format:
        IS_AUTHOR_QUERY: [YES/NO]
        AUTHOR_NAME: [author name if detected, otherwise NONE]

        Guidelines:
        - Answer YES if the query is asking about papers, research, or work by a specific author
        - Answer NO if it's a general topic query not focused on specific authors
        - For AUTHOR_NAME, extract the main author mentioned, or write NONE if no specific author
        - If multiple authors are mentioned, return all of them as a comma-separated 

        Examples:
        Query: "papers by John Smith"
        IS_AUTHOR_QUERY: YES
        AUTHOR_NAME: John Smith

        Query: "what is machine learning"
        IS_AUTHOR_QUERY: NO
        AUTHOR_NAME: NONE
        
        Query: "research on quantum computing by Alice Johnson and Bob Lee"
        IS_AUTHOR_QUERY: YES
        AUTHOR_NAME: Alice Johnson, Bob Lee
        
        Query: "latest advancements in AI"
        IS_AUTHOR_QUERY: NO
        AUTHOR_NAME: NONE
        
        Query: "Papers by Mark and Jane Doe"
        IS_AUTHOR_QUERY: YES
        AUTHOR_NAME: Mark, Jane Doe
        

        Query: "Einstein's work on relativity"
        IS_AUTHOR_QUERY: YES
        AUTHOR_NAME: Einstein

        Now analyze:
        """
        return PromptTemplate(
            template=template,
            input_variables=['query']
        )

    def detect_author_query_simple(self, query: str) -> tuple[int, List[str]]:
        """Simple LLM-based author detection
        Args:
            query (str): user query to analyze
        Returns:
            tuple[int, str]: (1, author_name) if author query detected, (0, "") if not
        """
        try:
            # Get the prompt template
            prompt = self.author_detector_prompt_simple()

            # Format and invoke LLM
            formatted_prompt = prompt.format(query=query)
            response = self.llm.invoke(formatted_prompt)

            # Extract content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Parse the response
            is_author_query = 0
            author_names = []

            lines = response_text.strip().split('\n')
            for line in lines:
                if line.startswith('IS_AUTHOR_QUERY:'):
                    answer = line.split(':', 1)[1].strip().upper()
                    is_author_query = 1 if answer == 'YES' else 0
                elif line.startswith('AUTHOR_NAME:'):
                    author_name = line.split(':', 1)[1].strip()
                    if author_name.upper() == 'NONE':
                        author_names = []
                    else:
                        author_names = [name.strip() for name in author_name.split(',')]

            logger.info(f"Author detection result: is_author={is_author_query}, author='{author_names}'")
            return is_author_query, author_names

        except Exception as e:
            logger.error(f"Error in simple LLM author detection: {e}")
            return 0, []

    def generate_answer(self, query: str) -> str:
        """Generate an answer using RAG approach
        Args:
            query: query to search for
        Returns:
            str: generated answer
        """
        try:
            # Use LLM to detect author queries
            is_author_query, author_name = self.detect_author_query_simple(query)
            logger.info(f"Author query detection: {is_author_query}, author name: {author_name}")

            if is_author_query:
                logger.info(f"Author query detected for: {author_name}")
                # Modify the search strategy for author queries
                k_value = self.DEFAULT_SEARCH_RESULTS * 2

            else:
                k_value = self.DEFAULT_SEARCH_RESULTS

            # Get context from vector search
            retrieved_results = self.perform_vectorstore_search(query=query, is_author_query=bool(is_author_query),
                                                                author_names=author_name, k=k_value)
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

            logger.info(f"Prompt formatted: {formatted_prompt}")

            # Invoke LLM with formatted prompt
            response = self.llm.invoke(formatted_prompt)

            # Return content based on a response type
            if hasattr(response, 'content'):
                response = response.content
            else:
                response = str(response)

            references = response.split("References:")[1].strip()
            formatted_references = self.format_references(references)

            answer = response.split("References:")[0].strip()
            final_answer = f"{answer}\n\n{formatted_references}"

            return final_answer


        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Sorry, I encountered an error while processing your query: {str(e)}"

    def format_references(self, answer: str) -> str:
        """Format the references in the answer
        Args:
            answer (str): generated answer
        Returns:
            str: formatted answer with references
        """
        if not answer:
            return "No references found."

        # Split by pattern that looks like "] [number]"
        references = re.split(r']\s*\[(?=\d)', answer)

        # Clean up the results
        cleaned_refs = []
        for i, ref in enumerate(references):
            if i == 0:
                # First reference already has opening bracket
                cleaned_refs.append(ref)
            else:
                # Add opening bracket to subsequent references
                cleaned_refs.append('[' + ref + ']')

        return '\n'.join(cleaned_refs)

# if __name__ == '__main__':
#     from src.models.embedding_engine import DocumentProcessor
#
#     qa_engine = QAEngine(llm='huggingface', doc_processor=DocumentProcessor())
