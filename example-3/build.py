#!.venv/bin/python

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from time import time
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import constants
from src.books import Book
from src.db import create_vector_store
from src.logger import logger


def get_pdf_loader(url: str) -> PyPDFLoader:
    return PyPDFLoader(
        file_path=url,
        headers={"User-Agent": "Mozilla/5.0"}
        # password = "my-password",
        # extract_images = True,
        # extraction_mode = "plain",
        # extraction_kwargs = None,
    )


def populate_vector_store(vector_store: Chroma, book: Book):
    start = time()
    logger.info(f"[{book.id}] Loading book: {book.name}")

    # Creating PDF loader
    loader = get_pdf_loader(book.url)

    # Loading docs from the loader
    docs = loader.load()
    logger.info(f"[{book.id}] Number of pages loaded: {len(docs)}")

    # Splitting pages
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=constants.CHUNK_SIZE,
        chunk_overlap=constants.CHUNK_OVERLAP,
    )
    all_splits = text_splitter.split_documents(docs)
    logger.info(f"[{book.id}] Number of splits: {len(all_splits)}")

    # Adding metadata to documents
    for doc in all_splits:
        doc.metadata.update({
            "id": str(uuid4()),
            "book.id": book.id,
            "book.name": book.name,
        })

    # Adding documents to the vector store
    vector_store.add_documents(documents=all_splits, ids=[doc.metadata["id"] for doc in all_splits])
    logger.info(f"[{book.id}] Added all documents in {time() - start:.2f} seconds!")


if __name__ == "__main__":
    start = time()

    # Loading books json
    books = Book.load(constants.BOOKS_JSON_PATH)

    # Creating vector store
    vector_store = create_vector_store(
        constants.EMBEDDINGS_MODEL_NAME,
        constants.COLLECTION_NAME,
        constants.VECTOR_STORE_DIR,
    )

    # Populating vector store with PDF files' content
    futures = deque()
    with ThreadPoolExecutor(max_workers=min(constants.MAX_THREADS, len(books))) as executor:
        for book in books:
            future = executor.submit(populate_vector_store, vector_store, book)
            futures.append(future)

    # Waiting for all the threads to complete processing
    while futures:
        future = futures.popleft()
        if future.running():
            futures.append(future)

    logger.info(f"Created vector store! Total time taken: {time() - start:.2f} seconds!")
