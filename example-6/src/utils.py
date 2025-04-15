import logging
import os
from typing import Any, List, LiteralString

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase

from . import prompts

logger = logging.getLogger(__name__)


def execute_cypher(query: LiteralString, **params) -> List[List[Any]]:
    logger.debug(f"Executing query: {query}")
    driver = GraphDatabase.driver(
        uri=os.getenv("NEO4j_URI"),
        auth=(os.getenv("NEO4j_AUTH_USER"), os.getenv("NEO4j_AUTH_PASSWORD")),
    )
    with driver.session() as session:
        result = session.run(query, **params)
        return result.values()


def recursive_shortening(batches: List[str], limit=10) -> List[str]:
    # Base case: if already under limit, return as is
    if len(batches) <= limit:
        return batches

    # Calculate chunk size based on number of batches and limit
    # This creates fewer, larger chunks when far over limit, and more balanced chunks when closer
    total_items = len(batches)
    num_chunks = max(2, min(total_items // 2, (total_items + limit - 1) // limit))
    chunk_size = (total_items + num_chunks - 1) // num_chunks

    # Create chunks of variable size
    sub_batches = []
    for i in range(0, total_items, chunk_size):
        sub_batches.append(batches[i:i + chunk_size])

    shortened_sub_batches = []
    for sub_batch in sub_batches:
        # Shortening sub batch
        llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
        parser = StrOutputParser()
        chain = prompts.RECURSIVE_SHORTENING_PROMPT | llm | parser
        shortened_sub_batch_content = chain.invoke({"descriptions": "\n".join(sub_batch)})
        shortened_sub_batches.append(shortened_sub_batch_content)

    # Recursively shortening batches until limit reached
    return recursive_shortening(shortened_sub_batches, limit)
