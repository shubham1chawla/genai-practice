import json
import logging
import os
from typing import List, Sequence

import requests
from llama_index.core import Document, Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(os.path.basename(__file__))

# This (Q4_K_M) takes about ~18 hours (~17s/it) on a Macbook M1 pro (16gb) for the Q/A extractor to run
LLM_MODEL_NAME = "llama3.1:8b"

# This (Q2_K) takes about ~13 hours (~13s/it) on a Macbook M1 pro (16gb) for the Q/A extractor to run
# LLM_MODEL_NAME = "llama3.1:8b-instruct-q2_K"

EMBEDDINGS_MODEL_NAME = "nomic-embed-text:latest"

BOOKS_JSON = os.path.join(os.path.dirname(__file__), "books.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "kv_store")


def download_books() -> List[dict]:
    with open(BOOKS_JSON, "r") as file:
        books = json.load(file)

    for book in books:
        if os.path.exists(os.path.join(DATA_DIR, f"{book["id"]}.pdf")):
            logger.info(f"[{book["id"]}] Already downloaded")
            continue

        # Creating directory
        if not os.path.exists(DATA_DIR) or not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)

        logger.info(f"[{book["id"]}] Downloading book...")
        with open(os.path.join(DATA_DIR, f"{book["id"]}.pdf"), "wb") as file:
            response = requests.get(book["url"])
            file.write(response.content)

    return books


def load_documents() -> List[Document]:
    books = download_books()
    books_dict = {book["id"]: book for book in books}

    def metadata_provider(file_path):
        book_id = os.path.basename(file_path).replace(".pdf", "")
        return {
            "book_name": books_dict[book_id]["name"],
            "file_path": file_path,
        }

    documents = SimpleDirectoryReader(DATA_DIR, file_metadata=metadata_provider).load_data()
    logger.info(f"Loaded {len(documents)} documents from data directory!")

    # Updating metadata of documents
    for document in documents:
        document.text_template = "Metadata:\n{metadata_str}\n-----\nContent:\n{content}"

        # Purging embeddings and llm metadata
        for key in ["page_label", "file_path"]:
            if key not in document.excluded_embed_metadata_keys:
                document.excluded_embed_metadata_keys.append(key)
            if key not in document.excluded_llm_metadata_keys:
                document.excluded_llm_metadata_keys.append(key)

    logger.info("Updated documents metadata!")
    return documents


def prepare_nodes() -> Sequence[BaseNode]:
    documents = load_documents()

    # Splitting text and generating question answers for better retrieval
    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(separator=" ", chunk_size=1024, chunk_overlap=128),

        # Comment the Q/A extractor to speed up the process of building the index
        QuestionsAnsweredExtractor(questions=2),
    ])

    nodes = pipeline.run(documents=documents, in_place=True, show_progress=True)
    logger.info(f"Prepared {len(nodes)} nodes for ingestion!")
    return nodes


def build_index():
    logger.info("Building VectorStoreIndex...")

    nodes = prepare_nodes()
    index = VectorStoreIndex(nodes=nodes, show_progress=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)

    logger.info("Built & saved the index!")


def main():
    # Setting up llama index
    Settings.llm = Ollama(model=LLM_MODEL_NAME, request_timeout=180, context_window=4096)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDINGS_MODEL_NAME)

    build_index()


if __name__ == "__main__":
    main()
