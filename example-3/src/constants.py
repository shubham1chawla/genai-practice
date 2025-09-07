from os import path

LLM_MODEL_NAME = "llama3.1:8b"
EMBEDDINGS_MODEL_NAME = "nomic-embed-text"
COLLECTION_NAME = "example-3"
VECTOR_STORE_DIR = path.join(path.dirname(__file__), "..", "./db")
BOOKS_JSON_PATH = path.join(path.dirname(__file__), "..", "./books.json")
CHUNK_SIZE = 256
CHUNK_OVERLAP = 16
MAX_THREADS = 8
K_VALUE = 5
REFINE_COUNT = 2