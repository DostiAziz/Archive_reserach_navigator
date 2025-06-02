import os
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
        'papers_data': "all",
        'vectorstore_ready': False,
        'processor': None,
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


def display_header():
    """Display the main application header"""
    st.markdown('<h1 class="main-header">🔬 Your AI Research Paper Navigator</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: #333; margin: 0;">🚀 Discover and explore research papers using advanced AI semantic search</h4>
            <p style="color: #666; margin: 0.5rem 0 0 0;">Powered by LangChain, ChromaDB, and Transformer models</p>
        </div>
        """, unsafe_allow_html=True)


def display_system_status():
    """Display current system status"""
    st.subheader("📊 System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        papers_count = len(st.session_state.papers_data) if st.session_state.papers_data is not None else 0
        status_class = "status-ready" if papers_count > 0 else "status-pending"

        st.markdown(f"""
        <div class="status-card {status_class}">
            <h4>📄 Paper Collection</h4>
            <h2>{papers_count}</h2>
            <p>{'✅ Ready' if papers_count > 0 else '⏳ No papers collected'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        vector_status = "✅ Ready" if st.session_state.vectorstore_ready else "❌ Not Ready"
        status_class = "status-ready" if st.session_state.vectorstore_ready else "status-pending"

        st.markdown(f"""
        <div class="status-card {status_class}">
            <h4>🧠 Vector Database</h4>
            <h3>{vector_status}</h3>
            <p>{'Search enabled' if st.session_state.vectorstore_ready else 'Collection needed'}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        chat_status = "✅ Available" if st.session_state.vectorstore_ready else "⏳ Pending"
        status_class = "status-ready" if st.session_state.vectorstore_ready else "status-pending"

        st.markdown(f"""
        <div class="status-card {status_class}">
            <h4>💬 Chat Interface</h4>
            <h3>{chat_status}</h3>
            <p>{'Ready to chat' if st.session_state.vectorstore_ready else 'Collection required'}</p>
        </div>
        """, unsafe_allow_html=True)


def display_sidebar():
    """Configure and display the sidebar with input controls"""
    st.sidebar.title("🎛️ Search Configuration")
    st.sidebar.markdown("---")

    # Search parameters
    st.sidebar.subheader("📝 Search Parameters")
    query = st.sidebar.text_input(
        "Research Query, for multi query separate them by **,** ",
        placeholder="e.g., machine learning, transformers, computer vision",
        help="Enter keywords or topics you want to research",
        key="search_query"
    )

    # Quick suggestion buttons
    st.sidebar.markdown("**💡 Quick Suggestions:**")
    suggestion_cols = st.sidebar.columns(2)

    suggestions = [
        "machine learning, deep learning", "transformers", "computer vision",
        "NLP", "AI ethics", "RAG"
    ]

    for i, suggestion in enumerate(suggestions):
        col = suggestion_cols[i % 2]
        if col.button(suggestion, key=f"suggestion_{i}"):
            # Use st.query_params or a different approach to handle the suggestion
            # Instead of directly setting session state, we'll return a flag
            st.session_state[f'suggestion_clicked_{i}'] = suggestion
            st.rerun()

    # Check if any suggestion was clicked and update search query
    for i, suggestion in enumerate(suggestions):
        if st.session_state.get(f'suggestion_clicked_{i}'):
            if st.session_state.search_query != suggestion:
                st.session_state.search_query = suggestion
            # Clear the flag
            st.session_state[f'suggestion_clicked_{i}'] = None

    # Number of documents
    st.sidebar.subheader("📊 Collection Size")
    num_docs = st.sidebar.slider(
        "Number of Papers",
        min_value=10,
        max_value=1000,
        value=50,
        step=10,
        help="More papers = better knowledge base but slower processing"
    )

    # Category selection
    st.sidebar.subheader("📂 Category Filter")
    categories = {
        "All Categories": 'all',
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
    st.sidebar.subheader("📊 Sort Order")
    sort_orders = {
        "Relevance": "relevance",
        "Most Recent": "lastUpdatedDate",
    }

    selected_sort = st.sidebar.selectbox(
        "Sort Results By",
        options=list(sort_orders.keys())
    )

    # Search history
    if st.session_state.search_history:
        st.sidebar.subheader("🕐 Recent Searches")
        for i, search in enumerate(st.session_state.search_history[-3:]):
            if st.sidebar.button(f"🔄 {search['query'][:20]}...", key=f"history_{i}"):
                st.session_state[f'history_clicked_{i}'] = search['query']
                st.rerun()

    # Check if any history item was clicked
    for i, search in enumerate(st.session_state.search_history[-3:]):
        if st.session_state.get(f'history_clicked_{i}'):
            if st.session_state.search_query != st.session_state[f'history_clicked_{i}']:
                st.session_state.search_query = st.session_state[f'history_clicked_{i}']
            # Clear the flag
            st.session_state[f'history_clicked_{i}'] = None

    return {
        'query': query,
        'num_docs': num_docs,
        'category': categories[selected_category],
        'sort_order': sort_orders[selected_sort],
    }


def display_progress_step(step_num: int, title: str, status: str, details: str):
    """Display the progress step with status
    Args:
        :param step_num (int) number of the current progress step
        :param title (str) title of the progress step
        :param status (str) status of the progress step
        :param details (str) details of the progress step
    """
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if status == 'completed':
            st.success("✅")
        if status == 'processing':
            st.warning("⏳")
        elif status == "error":
            st.error("❌")
        else:
            st.info("⏸️")

    with col2:
        st.write(f"**step {step_num}: {title}**")
        if details:
            st.caption(details)

    with col3:
        if status == 'completed':
            st.markdown("**Completed**")
        elif status == "processing":
            st.markdown("**Processing...**")
        elif status == "error":
            st.markdown("**Error**")
        else:
            st.markdown("Pending")


def collect_papers_with_parameter(params: Dict):
    """Using the submitted parameter for collecting papers
    Args:
        :params (Dict): A dictionary containing all the submitted parameters

    """
    st.subheader("📚 Building Your Research Database Using collected papers")

    # progress tracking
    progress_container = st.container()
    with progress_container:
        try:
            display_progress_step(1, "Initialize connection with Archive endpoint", "processing", "Setting up..")
            loader = DataPipeline()
            papers_data = loader.search_paper(query=params["query"],
                                              category=params["category"],
                                              max_results=params["num_docs"],
                                              sort_by=params["sort_order"]
                                              )

            if not papers_data:
                display_progress_step(2, "Search papers", "Error", "No papers found")
                st.error("❌ No papers found. Try different keywords.")
                return None

            display_progress_step(2, "Search papers", "Completed", f"Found {len(papers_data)} papers")

            # step 3: process documents
            display_progress_step(3, "Processing papers", "processing", "Converting to required format")
            papers_df = pd.DataFrame(data=papers_data)
            doc_processor = DocumentProcessor()
            documents = doc_processor.prepare_documents(papers_df)
            # uncomment this line if you want to chunk the documents
            # document_chunks = doc_processor.chunk_documents(documents)
            display_progress_step(3, "Processing papers", "Completed", f"Created {len(documents)} documents")

            # step 4 build vector databases
            display_progress_step(4, "Building your Research Database", "processing", "Create embeddings")
            collection_name = f"papers_{int(time.time())}"
            doc_processor.build_vectorstore(documents, collection_name)
            doc_processor.load_vectorstore(collection_name)
            display_progress_step(4, "Build Vector Database", "Completed", "Database is ready :)")

            st.session_state.papers_data = papers_df
            st.session_state.processor = doc_processor
            st.session_state.qa_engine = QAEngine(llm='genai', doc_processor=DocumentProcessor(),
                                                  vs_instance_name=collection_name)
            st.session_state.collection_name = collection_name
            st.session_state.vectorstore_ready = True

            # Add to search history
            st.session_state.search_history.append({
                'query': params['query'],
                'timestamp': datetime.now(),
                'num_papers': len(papers_data)
            })

            st.success("🎉 Collection completed successfully!")
            return papers_df

        except Exception as e:
            st.error(f"❌ Error during collection: {str(e)}")
            return None


def display_navigation_cards():
    """Display navigation cards to other pages"""
    st.subheader("🧭 Navigate to Other Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Analytics Dashboard</h3>
            <p>Explore comprehensive analytics of your collected papers including:</p>
            <ul>
                <li>📈 Category distributions</li>
                <li>📅 Publication timelines</li>
                <li>👥 Author networks</li>
                <li>🔍 Search and filter tools</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Explore Analytics", key="nav_analytics", use_container_width=True):
            st.switch_page("pages/analytics.py")

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 Research Chat</h3>
            <p>Interactive Q&A with your research collection:</p>
            <ul>
                <li>🤖 AI-powered responses</li>
                <li>📚 Source citations</li>
                <li>💡 Smart suggestions</li>
                <li>🔍 Semantic search</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        chat_disabled = not st.session_state.vectorstore_ready
        if st.button("💬 Start Chatting", key="nav_chat", use_container_width=True, disabled=chat_disabled):
            st.switch_page("pages/chat.py")

        if chat_disabled:
            st.caption("⚠️ Collect papers first to enable chat")


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()

    # Display header
    display_header()

    # Display system status
    display_system_status()

    # Get sidebar parameters
    params = display_sidebar()

    if not params:  # Module not available
        st.error("❌ Cannot proceed without required modules. Please check your installation.")
        return

    # Main content area
    if st.session_state.papers_data is None:
        # First time user - show welcome and collection interface
        st.markdown("---")
        st.subheader("🚀 Get Started with Your Research Journey")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **Welcome to your AI Research Navigator!** 

            Transform how you discover and explore academic papers:

            **📋 Step-by-step process:**
            1. 📝 **Enter your research query** in the sidebar (e.g., "machine learning", "computer vision")
            2. 📂 **Select relevant categories** to focus your search  
            3. 📊 **Choose collection size** (start with 30-50 papers)
            4. 🚀 **Click "Start Collection"** below to begin

            **🔄 What happens next:**
            - We'll search arXiv for relevant papers
            - Process and analyze the content  
            - Build an AI-powered search database
            - Enable chat and analytics features
            """)

        with col2:
            st.info("""
            **💡 Pro Tips:**

            **🎯 Search Strategy:**
            - Use specific keywords
            - Try different category combinations
            - Start with smaller collections (30-50 papers)

            **⚡ Performance:**
            - Fewer papers = faster processing
            - More papers = richer knowledge base

            **🔍 Examples:**
            - "transformer architecture"
            - "computer vision, deep learning"
            - "natural language processing"
            """)

        # Collection button
        st.markdown("---")

        if params['query'].strip():
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button(
                        "🚀 Start Paper Collection",
                        type="primary",
                        use_container_width=True,
                        disabled=st.session_state.collection_in_progress
                ):
                    if st.session_state.collection_in_progress:
                        st.warning("⏳ Collection already in progress...")
                    else:
                        with st.spinner("🔄 Initializing collection process..."):
                            result = collect_papers_with_parameter(params)

                        if result is not None:
                            time.sleep(1)  # Brief pause to show success
                            st.rerun()

                if st.session_state.collection_in_progress:
                    st.caption("⏳ Collection in progress... Please wait.")
        else:
            st.warning("⚠️ Please enter a research query in the sidebar to begin your exploration.")

    else:
        # User has data - show stats and navigation
        st.markdown("---")
        display_navigation_cards()

        # Option to collect new papers
        st.markdown("---")
        st.subheader("🔄 Start Fresh or Expand Collection")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            **Ready for a new research topic?** 

            Start a new collection to explore different papers, or modify your search parameters 
            to discover related research in your field.
            """)

        with col2:
            if st.button("🆕 New Collection", use_container_width=True, type="secondary"):
                # Reset session state for new collection (excluding widget-bound keys)
                keys_to_reset = [
                    'papers_data', 'vectorstore_ready', 'processor', 'collection_name',
                    'search_performed', 'current_step', 'collection_in_progress',
                    'last_search_params', 'chat_messages'
                    # Note: 'search_query' is excluded because it's bound to a widget
                ]
                for key in keys_to_reset:
                    if key in st.session_state:
                        if key == 'papers_data':
                            st.session_state[key] = None
                        elif key in ['vectorstore_ready', 'search_performed', 'collection_in_progress']:
                            st.session_state[key] = False
                        elif key in ['current_step']:
                            st.session_state[key] = 0
                        elif key in ['processor', 'collection_name', 'last_search_params']:
                            st.session_state[key] = None
                        elif key in ['chat_messages']:
                            st.session_state[key] = []

                # Show a message to clear the search query manually
                st.info("💡 Please clear the search query in the sidebar to start fresh!")
                st.rerun()


if __name__ == "__main__":
    import os
    setup_logging(log_level=os.getenv('LOG_LEVEL', 'INFO'))

    main()
