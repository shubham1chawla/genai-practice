import os

from dotenv import load_dotenv


load_dotenv()

BOOKS_JSON_PATH = './books.json'
EXPORT_DIR = './debug/'
CHUNK_SIZE = 800
CHUNK_OVERLAP = 20
LLM_MODEL_NAME = 'llama3.1:8b'
NEO4J_URI = 'bolt://localhost:7687'
NEO4J_AUTH = ('neo4j', os.environ['NEO4J_PASSWORD'])
MAX_THREADS = 8
