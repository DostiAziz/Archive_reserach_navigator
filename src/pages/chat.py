import streamlit as st
from typing import List, Dict

st.set_page_config(page_title="Chat with LLM",
                   layout="wide",
                   page_icon="💬")

st.title("Ask question regarding papers")

if not st.session_state.get('vectorstore_ready', False):
    st.warning("Vector database is not initialized yet. Please collect paper and try again.")
    if st.button("Main Page"):
        st.switch_page("streamlit_app.py")
    st.stop()


def initialize_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = " "


initialize_chat_state()


def display_chat_interface():
    st.subheader("Conversation history")

    chat_container = st.container()

    with chat_container:
        for i, message in enumerate(st.session_state.chat_messages):
            if message["role"] == 'user':
                with st.chat_message('user'):
                    st.write(message["content"])
            else:
                with st.chat_message('assistant'):
                    st.write(message["content"])
                    if 'source' in message:
                        with st.expander("Sources"):
                            for source in message['source']:
                                st.write(source)
    st.markdown('----')

    col1, col2 = st.columns([4,1])

    with col1:
        user_input = st.text_input(
            "Ask about your research papers:",
            placeholder="e.g., What are the main themes in these papers?",
            key="chat_input_field"
        )

        with col2:
            send_button = st.button("Send 📤", use_container_width=True)

    if send_button and user_input:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,

        })

        with st.spinner("Thinking..."):
            response = get_ai_response(user_input)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response,
            "source": response,
        })

        st.session_state.chat_input_field = " "
        st.experimental_rerun()


def get_ai_response(question: str) -> Dict:
    """Get AI response using your Q&A engine"""
    try:
        # Use your existing processor to query the vectorstore
        processor = st.session_state.processor

        if processor and processor.qa_chain:
            result = processor.qa_chain.invoke({"query": question})

            return {
                'answer': result.get('result', 'Sorry, I could not find a relevant answer.'),
                'sources': [doc.metadata.get('title', 'Unknown') for doc in result.get('source_documents', [])]
            }
        else:
            return {
                'answer': "Sorry, the Q&A system is not properly initialized.",
                'sources': []
            }
    except Exception as e:
        return {
            'answer': f"Error processing your question: {str(e)}",
            'sources': []
        }


# Display the chat interface
display_chat_interface()

# Sidebar info
st.sidebar.title("💬 Chat Settings")
st.sidebar.info(f"""
**Collection Info:**
- Papers: {len(st.session_state.papers_data) if st.session_state.papers_data is not None else 0}
- Vectorstore: {'✅ Ready' if st.session_state.vectorstore_ready else '❌ Not Ready'}
""")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_messages = []
    st.experimental_rerun()
