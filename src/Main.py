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
        'collection_name_input': ' ',
        'search_history': [],
        'last_search_params': None,
        'collection_in_progress': False,
        'search_query': ' '
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_header():
    """Display the main application header"""
    st.markdown('<h1 class="main-header">Your AI Research Paper Navigator</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: #333; margin: 0;">Discover and explore research papers in Archive</h4>
            <p style="color: #666; margin: 0.5rem 0 0 0;">Powered by LangChain, ChromaDB, and Streamlit</p>
        </div>
        """, unsafe_allow_html=True)


def display_system_status():
    """Display current system status"""
    st.markdown("-----------")
    st.subheader("📊 System Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        papers_count = len(st.session_state.papers_data) if st.session_state.papers_data is not None else 0
        status_class = "status-ready" if papers_count > 0 else "status-pending"

        st.markdown(f"""
        <div class="status-card {status_class}">
            <h4>📄 Collected papers</h4>
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
            <p>{'Search enabled' if st.session_state.vectorstore_ready else 'Collect papers'}</p>
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
    collection_name = st.sidebar.text_input("Collection Name",
                                            placeholder="Enter a collection name",
                                            help="Name that is used for your collected papers",
                                            value=st.session_state.collection_name)
    query = st.sidebar.text_input(
        "Research Query, for multi query separate them by **,** ",
        placeholder="e.g., machine learning, transformers, computer vision",
        help="Enter keywords or topics you want to research",
        key="search_query"
    )

    # Quick suggestion buttons
    st.sidebar.markdown("**💡 Quick Suggestions:**")
    suggestion_cols = st.sidebar.columns(2)

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

    return {
        'collection_name': collection_name,
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
        if status == 'Completed':
            st.success("✅")
        elif status == 'processing':
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
        if status == 'Completed':
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
            # Step 1: Initialize connection with Archive endpoint and perform search using query parameter
            display_progress_step(1, "Initialize connection with Archive endpoint", "processing", "Setting up..")
            loader = DataPipeline()
            # check if the query parameter contains ,
            if ',' in params['query']:
                papers_data = loader.list_of_queries(params['query'], category=params["category"],
                                                     max_results=params["num_docs"],
                                                     sort_by=params["sort_order"])
            else:
                papers_data = loader.search_paper(query=params["query"],
                                                  category=params["category"],
                                                  max_results=params["num_docs"],
                                                  sort_by=params["sort_order"]
                                                  )

            if not papers_data:
                display_progress_step(1, "Search papers", "Error", "No papers found")
                st.error("❌ No papers found. Try different keywords.")
                return None
            # Step 1: Show confirmation indicator
            display_progress_step(1, "Search papers", "Completed", f"Found {len(papers_data)} papers")
            time.sleep(2.0)

            # step 2: process documents
            display_progress_step(2, "Processing papers", "processing", "Converting to required format")
            papers_df = pd.DataFrame(data=papers_data)
            doc_processor = DocumentProcessor()
            documents = doc_processor.prepare_documents(papers_df)
            # uncomment this line if you want to chunk the documents
            # document_chunks = doc_processor.chunk_documents(documents)
            display_progress_step(2, "Processing papers", "Completed", f"Created {len(documents)} documents")
            time.sleep(2.0)

            # step 4 build vector databases
            collection_name = params['collection_name']
            display_progress_step(3, "Building your Research Database", "processing", "Create embeddings")
            doc_processor.build_vectorstore(documents, collection_name=collection_name)
            doc_processor.load_vectorstore(collection_name)
            display_progress_step(3, "Build Vector Database", "Completed", "Database is ready :)")

            st.session_state.papers_data = papers_df
            st.session_state.processor = doc_processor

            # Initializing chat endpoint
            display_progress_step(4, "Chat interface", "processing", "Initializing chat interface")
            st.session_state.qa_engine = QAEngine(llm='genai', doc_processor=doc_processor,
                                                  vs_instance_name=collection_name)
            st.session_state.collection_name = collection_name
            st.session_state.collection_name_input = collection_name
            st.session_state.vectorstore_ready = True

            # Add to search history
            st.session_state.search_history.append({
                'query': params['query'],
                'timestamp': datetime.now(),
                'num_papers': len(papers_data)
            })
            display_progress_step(4, "Chat interface", "Completed",
                                  "Chat interface is ready :), navigate toc chat page to start")
            time.sleep(1.0)
            st.success("🎉 Collection completed successfully!")
            time.sleep(2.0)
            return papers_df

        except Exception as e:
            st.error(f"❌ Error during collection: {str(e)}")
            return None


def display_navigation_cards():
    """Display navigation cards to other pages when data collection was successful"""

    st.subheader(" Navigate to other pages")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3> Analytics Dashboard</h3>
            <p>Explore visual analytics of your collected papers including:</p>
            <ul>
                <li> Category distribution</li>
                <li> Publication timelines</li>
                <li> Author networks</li>
                <li> Search and filter tools</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Explore visual analytics", key="nav_analytics", use_container_width=True):
            try:
                st.switch_page("pages/analytics.py")
            except Exception as e:
                st.info("An error occurred while nagivating to other pages")

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
            try:
                st.switch_page("pages/chat.py")
            except Exception as e:
                st.info(f"💬 Error while nagivating to Chat page {e}")

        if chat_disabled:
            st.caption("⚠️ Collect papers first to enable chat")


def display_quick_stats():
    """Display quick statistics if data is available"""
    if st.session_state.papers_data is not None:
        papers_data = st.session_state.papers_data

        st.markdown("### 📋 Collection Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📄 Papers", len(papers_data))

        with col2:
            # Extract all authors from all papers
            all_authors = []
            for authors_str in papers_data['authors']:
                if pd.notna(authors_str):
                    # Split by comma and clean up whitespace
                    authors = [author.strip() for author in str(authors_str).split(',')]
                    all_authors.extend(authors)

            # Count unique authors
            unique_authors = len(set(all_authors))
            st.metric("👥 Authors", unique_authors)

        with col3:
            # Count unique categories (simplified)
            categories = set()
            for categories_str in papers_data['categories']:
                if pd.notna(categories_str) and categories_str:
                    # Split by comma
                    cats = [cat.strip() for cat in str(categories_str).split(',')]
                    categories.update(cats)

            st.metric("📂 Categories", len(categories))

        with col4:
            if st.session_state.last_search_params:
                query = st.session_state.last_search_params['query']
                st.metric("🔍 Query", f"'{query[:15]}...'" if len(query) > 15 else f"'{query}'")


def reset_session_data():
    """Reset session data"""

    reset_values = {
        'papers_data': None,
        'vectorstore_ready': False,
        'doc_processor': None,
        'qa_engine': None,
        'collection_name': None,
        'search_performed': False,
        'current_step': 0,
        'collection_in_progress': False,
        'last_search_params': None,
        'chat_messages': [],
        'collection_name_input': "",
        'messages': []
    }

    # Reset each key to its default value
    for key, default_value in reset_values.items():
        if key in st.session_state:
            st.session_state[key] = default_value


def main():
    """Method to set up the whole interface"""

    # Initialize session state
    initialize_session_state()

    # Build the header part of the page
    display_header()

    # Display system status
    display_system_status()

    # get sidebar parameters
    params = display_sidebar()

    if not params:
        st.error("❌ Cannot proceed without required modules. Please check your installation.")
        return

    if st.session_state.papers_data is None:
        # First time user, show welcome and how the app can be used
        st.markdown("---")
        st.subheader("Get Started with Your AI Research Journey")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **Welcome to Your AI Research Navigator!**
            
            Transform how you discover and explore academic papers:
            
            **📋 Step-by-step process:**
            1. **Enter your search query** in the slidebar (e.g., Machine learning, Computer Vision, etc.)
             For multi query, separate query by **,**(e.g. RAG, agent)
            2. 📂 **Select relevant categories** to focus your search  
            3. 📊 **Choose collection size** (start with 30-50 papers)
            4. 🚀 **Click "Start Collection"** below to begin

            **🔄 What happens next:**
            - We'll search arXiv for relevant papers
            - Process and analyze the content  
            - Build an AI-powered search database
            - Enable chat and analytics features
            """
                        )

        with col2:
            st.info("""
            **💡Tips:**

            **🎯 Search Strategy:**
            - Use specific keywords
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
                if not st.session_state.collection_in_progress:
                    if st.button("Start paper collection",
                                 type="primary",
                                 use_container_width=True,
                                 disabled=False):
                        st.session_state.collection_in_progress = True
                        st.rerun()

                else:
                    # Show disabled button when in progress
                    st.button("🔄 Collection in Progress...",
                              type="secondary",
                              use_container_width=True,
                              disabled=True)
                if st.session_state.collection_in_progress:
                    with st.spinner("🔄 Initializing collection process..."):
                        result = collect_papers_with_parameter(params)
                        if result is not None:
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Collection process failed.")

        else:
            st.warning("Please enter a search query in the sidebar to begin collecting.")
    else:
        # Data collection is completed, now show the status and navigation
        st.markdown("---")
        display_quick_stats()

        st.markdown("---")
        display_navigation_cards()

        st.markdown("---")
        st.subheader("Start Fresh or Expand collection")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
                   **Ready for a new research topic?** 

                   Start a new collection to explore different papers, or modify your search parameters 
                   to discover related research in your field.
                   """)

        with col2:
            if st.button("🆕 New Collection", use_container_width=True, type="secondary"):
                # Reset session state for a new collection
                reset_session_data()
                st.rerun()


if __name__ == '__main__':
    setup_logging()
    main()
