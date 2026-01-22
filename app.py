# STREAMLIT UI


import streamlit as st
from rag_logic import build_vectorstore, build_rag_chain, answer_question

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot (Local, No OpenAI)")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


st.sidebar.header("Document Setup")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file and st.sidebar.button("Build Knowledge Base"):
    with st.spinner("Processing document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        vectorstore = build_vectorstore("temp.pdf")
        st.session_state.rag_chain = build_rag_chain(vectorstore)

        st.sidebar.success("RAG system ready")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.rag_chain:
        response = "Please upload a document first."
    else:
        chat_history = [
            m["content"]
            for m in st.session_state.messages
            if m["role"] == "assistant"
        ]

        with st.spinner("Thinking..."):
            response = answer_question(
                st.session_state.rag_chain,
                prompt,
                chat_history
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
