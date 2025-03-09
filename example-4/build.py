#!../.venv/bin/python


import logging
import os

from dotenv import load_dotenv

from src import graph


if __name__ == "__main__":
    load_dotenv()

    # Setting up logger
    logging.basicConfig(level=logging.INFO)
    for package in ['httpx', 'neo4j']:
        logging.getLogger(package).setLevel(logging.WARNING)

    # Invoking build graph
    graph.build_graph().invoke({
        'model': 'llama3.1:8b',                                   # Ollama Model to use
        'books_json_path': './harry-potter-books/books.json',     # Path to books.json file (Ex. Harry Potter)
        'export_dir': './harry-potter-books/exports',             # Export directory for saving LLM's output & metadata (Ex. Harry Potter)
        'chunk_size': 1000,                                       # Chunk size for text splitter
        'chunk_overlap': 0,                                       # Chunk overlap for text splitter
        'max_workers': 8,                                         # Parallel works to use. Remove or set 1 to use sequential processing
        'max_retries': 3,                                         # Max retries to address failed extractions
        'neo4j_uri': 'bolt://localhost:7687',                     # Neo4j database's URI
        'neo4j_auth': ('neo4j', os.environ['NEO4J_PASSWORD']),    # Neo4j database's username & password
        'neo4j_import_dir': './db/import'                         # Neo4j database's import directory (docker volume)
    })
