from models.qa_engine import QAEngine
from datetime import datetime
from typing import Dict
import pandas as pd
import time
import streamlit as st
from models.data_pipeline import DataPipeline
from models.embedding_engine import DocumentProcessor
from utils.logger_config import setup_logging

# Configure Streamlit page
st.set_page_config(
    page_title="Your AI Research Paper Navigator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .step-container {
        border: 2px solid #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #fafbfc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .status-success {
        color: #28a745;
        font-weight: bold;
    }

    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }

    .status-error {
        color: #dc3545;
        font-weight: bold;
    }

    .paper-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s ease;
    }

    .paper-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .paper-title {
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .paper-meta {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }

    .similarity-score {
        background-color: #e7f3ff;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    .progress-text {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'papers_data': None,
        'vectorstore_ready': False,
        'doc_processor': None,
        'qa_engine': None,
        'search_performed': False,
        'current_step': 0,
        'collection_name': None,
        'search_history': [],
        'last_search_params': None,
        'collection_in_progress': False,
        'chat_messages': []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

