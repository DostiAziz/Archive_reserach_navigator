import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Collection Analytics Dashboard")

# Check if data exists
if st.session_state.get('papers_data') is None:
    st.warning("⚠️ No data available. Please collect papers from the main page first.")
    if st.button("🏠 Go to Main Page"):
        st.switch_page("Main.py")
    st.stop()

papers_df = st.session_state.papers_data


# Analytics content
def display_analytics_dashboard(papers_df):
    """Display comprehensive analytics dashboard"""

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📄 Total Papers",
            value=len(papers_df),
            delta=f"+{len(papers_df)} collected"
        )

    with col2:
        unique_authors = len(set([author.strip() for authors in papers_df['authors']
                                  for author in authors.split(',')]))
        st.metric(
            label="👥 Unique Authors",
            value=unique_authors
        )

    with col3:
        categories = len(set([cat.strip() for cats in papers_df['categories']
                              for cat in cats.split(',')]))
        st.metric(
            label="📂 Categories",
            value=categories
        )

    with col4:
        papers_df['published'] = pd.to_datetime(papers_df['published'])
        recent_papers = len(papers_df[papers_df['published'] > '2022-01-01'])
        st.metric(
            label="🆕 Recent Papers > 2022-01-01",
            value=recent_papers,
            delta=f"{recent_papers / len(papers_df) * 100:.1f}% recent"
        )

# Run the analytics dashboard
display_analytics_dashboard(papers_df)
