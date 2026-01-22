from langchain_community.embeddings import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
