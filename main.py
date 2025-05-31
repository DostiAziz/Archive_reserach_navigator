#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from src.utils.logger_config import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging (this creates the log files)
setup_logging(log_level=os.getenv('LOG_LEVEL', 'INFO'))

# Get logger for main script
logger = get_logger('main')


def main():
    """Main application entry point"""
    logger.info("🚀 Starting Archive Research Navigator...")

    try:
        # Your app initialization here
        from src.models.qa_engine import QAEngine
        from src.models.embedding_engine import DocumentProcessor

        # Initialize components
        doc_processor = DocumentProcessor()
        qa_engine = QAEngine(llm='genai', doc_processor=doc_processor)

        logger.info("✅ Application started successfully")

        # Your app logic here...

    except Exception as e:
        logger.error(f"💥 Application failed to start: {str(e)}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())