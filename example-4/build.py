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
        'model': 'llama3.1:8b',
        'books_json_path': './books.json',
        'export_dir': './debug',
        'chunk_size': 1000,
        'chunk_overlap': 0,
        'max_workers': 8,
        'max_retries': 3,
        'neo4j_uri': 'bolt://localhost:7687',
        'neo4j_auth': ('neo4j', os.environ['NEO4J_PASSWORD']),
        'neo4j_import_dir': './db/import'
    })
