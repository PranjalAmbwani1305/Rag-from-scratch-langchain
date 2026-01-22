import streamlit as st
from rag_logic import load_split, build_store, build_chain, ask

st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if file and st.sidebar.button("Build"):
    with open("temp.pdf", "wb") as f:
        f.write(file.read())
    splits = load_split("temp.pdf")
    store = build_store(splits)
    st.session_state.chain = build_chain(store)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.chain:
        answer = "Upload a document first"
    else:
        history = [m["content"] for m in st.session_state.messages if m["role"] == "assistant"]
        answer = ask(st.session_state.chain, prompt, history)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
