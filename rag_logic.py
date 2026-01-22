from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Pinecone as PineconeStore
import pinecone

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.2,
    max_new_tokens=512,
)

def load_split(path):
    docs = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def build_store(splits):
    pinecone.init(
        api_key="YOUR_PINECONE_API_KEY",
        environment="YOUR_ENV"
    )

    index_name = "rag-course-index"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=384,
            metric="cosine"
        )

    return PineconeStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=index_name
    )

def docs_to_text(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template(
    "Answer only using the context below.\n\nContext:\n{context}\n\nQuestion:\n{question}"
)

def build_chain(store):
    retriever = store.as_retriever(search_kwargs={"k": 2})
    return (
        {
            "context": retriever | docs_to_text,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def rewrite(history, question):
    return llm.invoke(
        f"Rewrite the question as standalone.\nHistory:\n{history}\nQuestion:\n{question}"
    )

def ask(chain, question, history=None):
    if history:
        question = rewrite("\n".join(history), question)
    return chain.invoke(question)
