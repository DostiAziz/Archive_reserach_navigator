import streamlit as st
from typing import List, Dict

st.set_page_config(page_title="Chat with LLM",
                   layout="wide",
                   page_icon="💬")

# Custom CSS for fixed chat input at the bottom
st.markdown("""
<style>
    /* Hide default streamlit padding */
    .main .block-container {
        padding-bottom: 5rem;
    }
    
    /* Fixed input container at bottom */
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e0e0e0;
        padding: 1rem;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    /* Chat messages container with proper scrolling */
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 5rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background: #fafafa;
    }
    
    /* Style for individual messages */
    .user-message {
        background: #007bff;
        color: white;
        padding: 0.75rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .assistant-message {
        background: #f1f3f4;
        color: #333;
        padding: 0.75rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    
    /* Input styling */
    .chat-input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .chat-input:focus {
        border-color: #007bff;
        outline: none;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
    }
    
    /* Send button styling */
    .send-button {
        background: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 1.2rem;
    }
    
    .send-button:hover {
        background: #0056b3;
    }
    
    /* Hide streamlit elements we don't need */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .assistant-message {
            margin-left: 5%;
            margin-right: 5%;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 Ask questions regarding papers")

if not st.session_state.get('vectorstore_ready', False):
    st.warning("Vector database is not initialized yet. Please collect paper and try again.")
    if st.button("🏠 Main Page"):
        st.switch_page("../Main.py")
    st.stop()


def initialize_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = ""


initialize_chat_state()


def get_ai_response(question: str) -> Dict:
    """Get AI response using your Q&A engine"""
    try:

        processor = st.session_state.processor
        qa_engine = st.session_state.qa_engine
        if processor and qa_engine:
            result = qa_engine.generate_answer(question)

            return result
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


def display_chat_messages():
    """Display chat messages in a scrollable container"""
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Welcome to Research Chat!</h3>
                <p>Start a conversation by asking questions about your collected papers.</p>
                <p><em>Example: "What are the main themes in these papers?"</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display messages
            for i, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        
                        # Display sources if available
                        if "sources" in message and message["sources"]:
                            with st.expander("📚 Sources", expanded=False):
                                for idx, source in enumerate(message["sources"], 1):
                                    st.write(f"{idx}. {source}")


def handle_user_input():
    """Handle user input and generate response"""
    
    # Create the input area at the bottom
    st.markdown("---")
    
    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                label="Message",
                placeholder="Ask about your research papers...",
                key="user_message_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_clicked = st.form_submit_button("📤 Send", use_container_width=True)
        
        # Handle Enter key submission
        if send_clicked and user_input.strip():
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate AI response
            with st.spinner("🤔 Thinking..."):
                response = get_ai_response(user_input)
            
            # Add assistant response
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response,
                "sources": []
            })
            
            # Rerun to update the display
            st.rerun()


# Main chat interface
def main():
    # Display chat messages
    display_chat_messages()
    
    # Handle user input (this creates the fixed input at bottom)
    handle_user_input()
    
    # Sidebar info
    with st.sidebar:
        st.title("💬 Chat Settings")
        
        # Collection info
        st.info(f"""
        **📊 Collection Info:**
        - Papers: {len(st.session_state.papers_data) if st.session_state.papers_data is not None else 0}
        - Vectorstore: {'✅ Ready' if st.session_state.vectorstore_ready else '❌ Not Ready'}
        - Messages: {len(st.session_state.chat_messages)}
        """)
        
        st.markdown("---")
        
        # Chat controls
        st.subheader("🛠️ Chat Controls")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        
        if st.button("🏠 Back to Main", use_container_width=True):
            st.switch_page("../Main.py")
        
        st.markdown("---")
        
        # Quick suggestions
        st.subheader("💡 Quick Questions")
        quick_questions = [
            "What are the main themes?",
            "Summarize key findings",
            "Who are the top authors?",
            "What methodologies are used?",
            "Recent trends in research?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                # Add the question as user input
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": question
                })
                
                # Generate response
                with st.spinner("🤔 Thinking..."):
                    response = get_ai_response(question)
                
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": []
                })
                
                st.rerun()


# Run the app
if __name__ == "__main__":
    main()