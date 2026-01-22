from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)

def load_split(path):
    docs = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def build_store(splits):
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

def docs_to_text(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template(
    "Answer only using the context.\nContext:\n{context}\n\nQuestion:\n{question}"
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
        f"Rewrite as standalone question.\nHistory:\n{history}\nQuestion:\n{question}"
    )

def ask(chain, question, history=None):
    if history:
        question = rewrite("\n".join(history), question)
    return chain.invoke(question)
