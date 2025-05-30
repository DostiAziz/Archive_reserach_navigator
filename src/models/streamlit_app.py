import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import time
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go



# Configure Streamlit page
st.set_page_config(
    page_title="AI Research Paper Navigator",
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
        'processor': None,
        'search_performed': False,
        'current_step': 0,
        'collection_name': None,
        'search_history': [],
        'last_search_params': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_header():
    """Display the main application header"""
    st.markdown('<h1 class="main-header">🔬 AI Research Paper Navigator</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: #333; margin: 0;">🚀 Discover and explore research papers using advanced AI semantic search</h4>
            <p style="color: #666; margin: 0.5rem 0 0 0;">Powered by LangChain, ChromaDB, and Transformer models</p>
        </div>
        """, unsafe_allow_html=True)



def display_sidebar():
    """Configure and display the sidebar with input controls"""
    st.sidebar.title("🎛️ Search Configuration")
    st.sidebar.markdown("---")

    # Search parameters
    st.sidebar.subheader("📝 Search Parameters")
    query = st.sidebar.text_input(
        "Research Query",
        placeholder="e.g., machine learning, transformers, computer vision",
        help="Enter keywords or topics you want to research",
        key="search_query"
    )

    # Quick suggestion buttons
    st.sidebar.markdown("**💡 Quick Suggestions:**")
    suggestion_cols = st.sidebar.columns(2)

    suggestions = [
        "machine learning", "deep learning", "transformers", "computer vision",
        "NLP", "reinforcement learning", "neural networks", "AI ethics"
    ]

    for i, suggestion in enumerate(suggestions):
        col = suggestion_cols[i % 2]
        if col.button(suggestion, key=f"suggestion_{i}"):
            st.session_state.search_query = suggestion
            st.experimental_rerun()

    st.sidebar.markdown("---")

    # Number of documents
    st.sidebar.subheader("📊 Collection Size")
    num_docs = st.sidebar.slider(
        "Number of Papers",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="More papers = better knowledge base but slower processing"
    )

    # Category selection
    st.sidebar.subheader("📂 Category Filter")
    categories = {
        "All Categories": None,
        "🤖 Artificial Intelligence": "cs.AI",
        "🧠 Machine Learning": "cs.LG",
        "👁️ Computer Vision": "cs.CV",
        "💬 Natural Language Processing": "cs.CL",
        "🧬 Neural Networks": "cs.NE",
        "📊 Statistics - ML": "stat.ML",
        "🔢 Mathematics": "math.ST",
        "🏗️ Software Engineering": "cs.SE",
        "🔐 Cryptography": "cs.CR",
        "🌐 Human-Computer Interaction": "cs.HC"
    }

    selected_category = st.sidebar.selectbox(
        "Select Category",
        options=list(categories.keys()),
        help="Filter papers by arXiv category"
    )

    # Sort order
    st.sidebar.subheader("📊 Sort & Filter")
    sort_orders = {
        "📈 Most Relevant": "relevance",
        "🕐 Most Recent": "lastUpdatedDate",
        "📚 Submission Date": "submittedDate"
    }

    selected_sort = st.sidebar.selectbox(
        "Sort Results By",
        options=list(sort_orders.keys())
    )

    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        chunk_size = st.slider("Document Chunk Size", 500, 2000, 1000, 100,
                               help="Smaller chunks = more precise but more processing")
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.1, 0.05,
                                         help="Higher threshold = more strict matching")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
            help="Different models have different strengths"
        )

    st.sidebar.markdown("---")

    # Search history
    if st.session_state.search_history:
        st.sidebar.subheader("🕐 Recent Searches")
        for i, search in enumerate(st.session_state.search_history[-3:]):
            if st.sidebar.button(f"🔄 {search['query'][:20]}...", key=f"history_{i}"):
                st.session_state.search_query = search['query']
                st.experimental_rerun()

    return {
        'query': query,
        'num_docs': num_docs,
        'category': categories[selected_category],
        'sort_order': sort_orders[selected_sort],
        'chunk_size': chunk_size,
        'similarity_threshold': similarity_threshold,
        'embedding_model': embedding_model
    }


def display_progress_tracker(steps: List[Dict]):
    """Display animated progress tracker"""
    st.subheader("📋 Collection Progress")

    progress_cols = st.columns(len(steps))
    progress_bar = st.progress(0)
    status_text = st.empty()

    completed_steps = sum(1 for step in steps if step['status'] == 'completed')
    progress_value = completed_steps / len(steps)
    progress_bar.progress(progress_value)

    for i, (col, step) in enumerate(zip(progress_cols, steps)):
        with col:
            if step['status'] == 'completed':
                st.success(f"✅ {step['title']}")
            elif step['status'] == 'processing':
                st.warning(f"⏳ {step['title']}")
                status_text.markdown(f"🔄 **Currently:** {step['details']}")
            elif step['status'] == 'error':
                st.error(f"❌ {step['title']}")
            else:
                st.info(f"⏸️ {step['title']}")

    return progress_bar, status_text


def collect_papers_with_progress(params: Dict):
    """Collect papers with real-time progress updates"""
    st.subheader("📚 Building Your Research Database")

    # Define steps
    steps = [
        {'title': 'Initialize', 'status': 'pending', 'details': ''},
        {'title': 'Search arXiv', 'status': 'pending', 'details': ''},
        {'title': 'Process Docs', 'status': 'pending', 'details': ''},
        {'title': 'Build Vector DB', 'status': 'pending', 'details': ''}
    ]

    progress_bar, status_text = display_progress_tracker(steps)

    try:
        # Step 1: Initialize
        steps[0].update({'status': 'processing', 'details': 'Setting up arXiv connection...'})
        display_progress_tracker(steps)
        time.sleep(0.5)  # Visual feedback

        loader = ArxivDataLoader()
        steps[0].update({'status': 'completed', 'details': 'arXiv loader ready'})
        display_progress_tracker(steps)

        # Step 2: Search Papers
        steps[1].update({'status': 'processing', 'details': f"Searching for '{params['query']}'..."})
        display_progress_tracker(steps)

        papers_data = loader.search_papers(
            query=params['query'],
            max_results=params['num_docs'],
            category=params['category']
        )

        if not papers_data:
            steps[1].update({'status': 'error', 'details': 'No papers found'})
            display_progress_tracker(steps)
            st.error("❌ No papers found for your query. Try different keywords or categories.")
            return None

        steps[1].update({'status': 'completed', 'details': f'Found {len(papers_data)} papers'})
        display_progress_tracker(steps)

        # Step 3: Process Documents
        steps[2].update({'status': 'processing', 'details': 'Converting to LangChain documents...'})
        display_progress_tracker(steps)

        papers_df = pd.DataFrame(papers_data)
        processor = DocumentProcessor(embedding_model=params['embedding_model'])
        st.session_state.processor = processor

        documents = processor.papers_to_documents(papers_df)
        split_docs = processor.split_documents(documents)

        steps[2].update({'status': 'completed', 'details': f'Created {len(split_docs)} chunks'})
        display_progress_tracker(steps)

        # Step 4: Build Vector Database
        steps[3].update({'status': 'processing', 'details': 'Creating embeddings and vector index...'})
        display_progress_tracker(steps)

        collection_name = f"papers_{int(time.time())}"
        processor.create_vectorstore(split_docs, collection_name=collection_name)

        steps[3].update({'status': 'completed', 'details': 'Vector database ready!'})
        display_progress_tracker(steps)

        # Update session state
        st.session_state.papers_data = papers_df
        st.session_state.vectorstore_ready = True
        st.session_state.collection_name = collection_name
        st.session_state.last_search_params = params

        # Add to search history
        st.session_state.search_history.append({
            'query': params['query'],
            'timestamp': datetime.now(),
            'num_papers': len(papers_data)
        })

        status_text.success("🎉 Collection completed successfully!")
        return papers_df

    except Exception as e:
        st.error(f"❌ Error during collection: {str(e)}")
        return None


def display_collection_analytics(papers_df: pd.DataFrame):
    """Display comprehensive analytics of the collected papers"""
    st.subheader("📊 Collection Analytics")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #1f77b4; margin: 0;">📄</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <p style="margin: 0; color: #666;">Total Papers</p>
        </div>
        """.format(len(papers_df)), unsafe_allow_html=True)

    with col2:
        unique_authors = len(set([author.strip() for authors in papers_df['authors']
                                  for author in authors.split(',')]))
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff7f0e; margin: 0;">👥</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <p style="margin: 0; color: #666;">Unique Authors</p>
        </div>
        """.format(unique_authors), unsafe_allow_html=True)

    with col3:
        categories = len(set([cat.strip() for cats in papers_df['categories']
                              for cat in cats.split(',')]))
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2ca02c; margin: 0;">📂</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <p style="margin: 0; color: #666;">Categories</p>
        </div>
        """.format(categories), unsafe_allow_html=True)

    with col4:
        papers_df['published'] = pd.to_datetime(papers_df['published'])
        recent_papers = len(papers_df[papers_df['published'] > '2022-01-01'])
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #d62728; margin: 0;">🆕</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <p style="margin: 0; color: #666;">Recent Papers</p>
        </div>
        """.format(recent_papers), unsafe_allow_html=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Papers by Category")
        category_data = []
        for cats in papers_df['categories']:
            category_data.extend([cat.strip() for cat in cats.split(',')])

        category_counts = pd.Series(category_data).value_counts().head(10)

        # fig = px.bar(
        #     x=category_counts.values,
        #     y=category_counts.index,
        #     orientation='h',
        #     title="Top 10 Categories",
        #     labels={'x': 'Number
