import streamlit as st

from myrag import LearningAssistant

# Initialize models and RAG
rag = LearningAssistant.get_instance()

# Streamlit app title
st.title("LLM Workshop Quiz")
st.write("""
Guide this chatbot to answer questions about LLMs that are available in the context.
Give a topic to the chat like "LLMs" or "rag", "chain of thought", "role models in ai", "ai ethics", etc
""")

# Initialize session state for quiz
# add the messages to the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# append to the chat history and display the conversations  
if prompt:= st.chat_input("Ask for a quiz question"):

# Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    response = rag.query(prompt)
    # remove any reference to the correct answer
    response = response.replace("(CORRECT)", "")
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
