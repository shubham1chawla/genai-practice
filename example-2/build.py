#!.venv/bin/python

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import os

MODEL_NAME = "llama3.2:3b"
COLLECTION_NAME = "example-2"
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "./db")
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
BOOK_URLS = [
   "https://www.usu.edu/aggiewellness/files/USU-Student-Cookbook-FINAL-1.pdf",
   "https://foodhero.org/sites/foodhero-prod/files/health-tools/cookbook.pdf",
   "https://theopendoorpantry.org/wp-content/uploads/2022/09/Recipe-Book.pdf",
   "https://www.i4n.in/wp-content/uploads/2023/05/Recipe-Book.pdf",
   "https://www.gourmia.com/pdf_recipes/GAF858-Recipe-Book.pdf",
   "https://extension.purdue.edu/foodlink/includes/pubs/recipebooklet.pdf",
   "https://www.papreferred.com/assets/en/recipe-book.pdf",
   "https://first.global/wp-content/uploads/2020/10/2020-FGC-Flavors-of-the-World-Cookbook_v1.2.pdf",
   "https://www.kidney.org/sites/default/files/docs/kidney_cookbook_lr.pdf",
]


def get_pdf_loader(url: str) -> PyPDFLoader:
    return PyPDFLoader(
        file_path = url,
        headers = {"User-Agent": "Mozilla/5.0"}
        # password = "my-password",
        # extract_images = True,
        # extraction_mode = "plain",
        # extraction_kwargs = None,
    )


def create_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )


def populate_vector_store(vector_store: Chroma, url: str):
    logger.info(f"Loading book from URL: {url}")

    # Creating PDF loader
    loader = get_pdf_loader(url)

    # Loading docs from the loader
    docs = loader.load()
    logger.info(f"Number of pages loaded: {len(docs)}")

    # Splitting pages
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,   
    )
    all_splits = text_splitter.split_documents(docs)
    logger.info(f"Number of splits: {len(all_splits)}")

    # Adding documents to the vector store
    vector_store.add_documents(documents=all_splits)
    logger.info("--------------------------------------------------")


if __name__ == "__main__":
    # Creating vector store
    vector_store = create_vector_store()

    # Populating vector store with PDF files' content
    for url in BOOK_URLS:
        populate_vector_store(vector_store, url)

    logger.info("Created vector store!")