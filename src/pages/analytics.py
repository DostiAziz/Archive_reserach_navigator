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
        st.switch_page("streamlit_app.py")
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
            label="🆕 Recent Papers",
            value=recent_papers,
            delta=f"{recent_papers / len(papers_df) * 100:.1f}% recent"
        )

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Papers by Category")
        category_data = []
        for cats in papers_df['categories']:
            category_data.extend([cat.strip() for cat in cats.split(',')])

        category_counts = pd.Series(category_data).value_counts().head(10)

        fig = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Top 10 Categories",
            labels={'x': 'Number of Papers', 'y': 'Category'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Publication Timeline")
        papers_df['year'] = papers_df['published'].dt.year
        yearly_counts = papers_df['year'].value_counts().sort_index()

        fig = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            title="Papers by Publication Year",
            labels={'x': 'Year', 'y': 'Number of Papers'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed tables
    st.subheader("📋 Paper Details")

    # Search and filter
    search_term = st.text_input("🔍 Search papers by title or authors:")
    if search_term:
        filtered_df = papers_df[
            papers_df['title'].str.contains(search_term, case=False, na=False) |
            papers_df['authors'].str.contains(search_term, case=False, na=False)
            ]
    else:
        filtered_df = papers_df

    # Display papers
    st.dataframe(
        filtered_df[['title', 'authors', 'published', 'categories']],
        use_container_width=True,
        height=400
    )


# Run the analytics dashboard
display_analytics_dashboard(papers_df)