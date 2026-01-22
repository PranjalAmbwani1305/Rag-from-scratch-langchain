import streamlit as st
import rag_logic

st.title("LangChain RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded and st.sidebar.button("Build Index"):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())

    splits = rag_logic.load_split("temp.pdf")
    store = rag_logic.build_store(splits)
    st.session_state.chain = rag_logic.build_chain(store)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.chain:
        answer = "Upload and index a document first."
    else:
        history = [m["content"] for m in st.session_state.messages if m["role"] == "assistant"]
        answer = rag_logic.ask(st.session_state.chain, prompt, history)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
