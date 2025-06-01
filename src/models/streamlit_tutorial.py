import streamlit as st

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
    st.markdown('<h1 class="main-header">🔬 Your AI Research Paper Navigator</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(""" Developed by Dosti""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: #333; margin: 0;">🚀 Discover and explore research papers using advanced AI semantic search</h4>
            <p style="color: #666; margin: 0.5rem 0 0 0;">Powered by LangChain, ChromaDB, and Transformer models</p>
        </div>
        """, unsafe_allow_html=True)


initialize_session_state()
display_header()


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
            st.session_state.search_query = suggestion
            st.experimental_rerun()

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





display_sidebar()
