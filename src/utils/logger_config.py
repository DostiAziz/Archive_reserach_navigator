import logging
import sys
from pathlib import Path
from datetime import datetime

from config import Config


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration
    Args:
        log_level (str, optional): Logging level. Defaults to "INFO".
    """
    # Convert string level to logging constant if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    log_dir = Path(Config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "app").mkdir(exist_ok=True)
    (log_dir / "errors").mkdir(exist_ok=True)

    # Get the current date for log files
    today = datetime.now().strftime("%Y-%m-%d")

    logging.getLogger().handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_dir / f"app/app_{today}.log")
    error_handler = logging.FileHandler(log_dir / f"errors/errors_{today}.log")
    error_handler.setLevel(logging.ERROR)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            console_handler,
            file_handler,
            error_handler,
        ]
    )

    # Print confirmation
    print(f"📝 Logging setup complete!")
    print(f"   - All logs: logs/app/app_{today}.log")
    print(f"   - Error logs: logs/errors/errors_{today}.log")

    loggers = {
        'qa_engine': logging.getLogger('qa_engine'),
        "data_pipeline": logging.getLogger('data_pipeline'),
        "embedding_engine": logging.getLogger('embedding_engine'),
        "main": logging.getLogger('main'),
        'streamlit': logging.getLogger('streamlit'),
    }
    return loggers


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""

    return logging.getLogger(name)
