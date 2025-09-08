import logging
import os

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(os.path.basename(__file__))

# Using standard (Q4_K_M) LLM to generate the answers for the user
LLM_MODEL_NAME = "llama3.1:8b"
EMBEDDINGS_MODEL_NAME = "nomic-embed-text:latest"

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "kv_store")


def load_index() -> VectorStoreIndex:
    if not os.path.exists(STORAGE_DIR):
        raise FileNotFoundError("Run `build.py` to build the VectorStoreIndex!")

    logger.info("Loading index from kv_store directory...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context=storage_context)

    if not isinstance(index, VectorStoreIndex):
        raise ValueError(f"Loaded index is not of type {VectorStoreIndex.__class__}")
    return index


def main():
    # Setting up llama index
    Settings.llm = Ollama(model=LLM_MODEL_NAME, request_timeout=180, context_window=2048)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDINGS_MODEL_NAME)

    # Querying index
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
    response = query_engine.query("What's the relationship between Harry and Dobby?")

    # Printing response
    print("-" * 100)
    print(response)
    print("-" * 100)
    print("Sources:")
    for i, metadata in enumerate(response.metadata.values()):
        print(f"{i + 1}. {metadata["book_name"]}")


if __name__ == "__main__":
    main()
