from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings


def create_vector_store(model_name: str, collection_name: str, dir: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=OllamaEmbeddings(model=model_name),
        persist_directory=dir,
    )