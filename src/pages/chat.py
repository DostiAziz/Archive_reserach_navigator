import streamlit as st

st.set_page_config(page_title="Chat with LLM",
                   layout="wide",
                   page_icon="💬")

st.title("💬 Ask questions regarding papers")

if not st.session_state.get('vectorstore_ready', False):
    st.warning("Vector database is not initialized yet. Please collect paper and try again.")
    if st.button("🏠 Main Page"):
        st.switch_page("Main.py")
    st.stop()

# Initialize messages if not exists
if "messages" not in st.session_state:
    st.markdown("""
                 <div style="text-align: center; padding: 2rem; color: #666;">
                    <h3>👋 Welcome to Research Chat!</h3>
                     <p>Start a conversation by asking questions about your collected papers.</p>
                     <p><em>Example: "What are the main themes in these papers?"</em></p>
                 </div>
                 """, unsafe_allow_html=True)
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def QA_chat():
    # Handle new user input
    if prompt := st.chat_input("Ask Your Questions"):
        # Add a user message to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        processor = st.session_state.processor
        qa_engine = st.session_state.qa_engine

        if processor and qa_engine:
            try:
                with st.spinner("🤔 AI is thinking..."):
                    result = qa_engine.generate_answer(prompt)

                # Display the result in a chat message
                with st.chat_message("assistant"):
                    st.markdown(result)

                # Store the result
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            error_msg = "❌ QA Engine not available. Please collect papers first."
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Main chat interface
def main():
    # Display chat Interface
    QA_chat()


# Run the app
if __name__ == "__main__":
    main()
