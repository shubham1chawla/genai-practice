import os

from dotenv import load_dotenv


load_dotenv()

BOOKS_JSON_PATH = './books.json'
EXPORT_DIR = './debug/'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
LLM_MODEL_NAME = 'llama3.1:8b'
NEO4J_URI = 'bolt://localhost:7687'
NEO4J_AUTH = ('neo4j', os.environ['NEO4J_PASSWORD'])
MAX_WORKERS = 8
