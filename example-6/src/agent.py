import logging

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from neo4j.graph import Node, Relationship

from . import prompts, utils

logger = logging.getLogger(__name__)


@tool
def get_books_by_query(query: str) -> str:
    """
    Finds related books that contains or are similar to the query.

    :param query: part of the name to find similar books
    :return: list of books matching the query
    :raises ValueError: If the input parameter query is empty
    """
    if not query:
        raise ValueError(f"You must provide a `query` to `{get_books_by_query.__name__}` tool!")

    rows = utils.execute_cypher(
        "MATCH (b:Book) WHERE b.name =~ $pattern RETURN b",
        pattern=f"(?i)(|.*.){".*.".join(query.split())}(|.*.)"
    )
    if not rows:
        return f"No books found similar to the query '{query}'!"

    logger.info(f"Found {len(rows)} book nodes matching query: '{query}'")
    books = []
    for i, values in enumerate(rows):
        node: Node = values[0]
        books.append(f"{i + 1}. {node.get("name", "unknown")}")

    return f"""Total: {len(rows)} books
{"\n".join(books)}
    """


@tool
def get_entities_by_query(entity_query: str, book_query: str = "") -> str:
    """
    Finds related entities (characters, locations, objects, etc.) that contains or are similar to the entity query.

    Entities contain following information:
    - Entity's name
    - Whether they are singular or plural
    - What type of entities are they
    - In which books they were mentioned
    - How are entities described in the books

    :param entity_query: part of the name or type to find similar entities
    :param book_query: optional, part of the name to filter entities based on specific books
    :return: list of entities matching the queries
    :raises ValueError: If the input parameter entity query is empty
    """
    if not entity_query:
        raise ValueError(f"You must provide a `entity_query` to `{get_entities_by_query.__name__}` tool!")

    rows = utils.execute_cypher(
        f"""
        MATCH (e:Entity)-[]-(d:Description)-[]-(b:Book)
        WHERE 
        (e.name =~ $entities_pattern OR ANY(type IN e.entity_types WHERE type =~ $entities_pattern))
        AND 
        b.name =~ $books_pattern
        return e, d, b
        """,
        entities_pattern=f"(?i)(|.*.){".*.".join(entity_query.split())}(|.*.)",
        books_pattern=f"(?i)(|.*.){".*.".join(book_query.split())}(|.*.)"
    )
    if not rows:
        return f"No entities found similar to the entity query '{entity_query}' and book query '{book_query}'!"

    # Creating a nested dictionary for rows to club information together
    nested_dict = {}
    for row in rows:
        entity_node: Node = row[0]
        description_node: Node = row[1]
        book_node: Node = row[2]
        description_content = description_node.get("content")

        if entity_node.id not in nested_dict:
            nested_dict[entity_node.id] = {
                "entity": entity_node,
                "books": {
                    book_node.id: {
                        "book": book_node,
                        "descriptions": {description_content} if description_content else {}
                    }
                }
            }
        elif book_node.id not in nested_dict[entity_node.id]["books"]:
            nested_dict[entity_node.id]["books"][book_node.id] = {
                "book": book_node,
                "descriptions": {description_content} if description_content else {}
            }
        elif description_content:
            nested_dict[entity_node.id]["books"][book_node.id]["descriptions"].add(description_content)

    logger.info(f"Found {len(nested_dict)} entities | entity: '{entity_query}' | book: '{book_query}'")

    infos = []
    for ei, entity_dict_value in enumerate(nested_dict.values()):
        logger.info(f"Creating info for entity: {ei + 1}")
        entity_node: Node = entity_dict_value["entity"]

        # Creating top-level entity info
        info = [
            f"{ei + 1}. {entity_node.get("name", "unknown")} ({"singular" if entity_node.get("singular", True) else "plural"})",
            f"- {", ".join(entity_node.get("entity_types", []))}",
            "",
        ]

        # Iterating over books in which the entity is mentioned
        for book_dict_value in entity_dict_value["books"].values():
            shortened_descriptions = utils.recursive_shortening(list(book_dict_value["descriptions"]))

            # Creating child-level book info and adding context
            book_node: Node = book_dict_value["book"]
            info += [f"{"\t" * 1}Mentioned in the book '{book_node.get("name", "unknown")}' as:", ""]
            info += [f"{"\t" * 2}- {shortened_description}" for shortened_description in shortened_descriptions]
            info += [""]

        infos.append("\n".join(info))

    return f"""Total: {len(nested_dict)} entities
{"\n".join(infos)}    
    """


@tool
def get_relationships_by_query(entity1_query: str, entity2_query: str) -> str:
    """
    Finds related relationships between entities (characters, locations, objects, etc.) that contains or are similar to
    the entity 1 and entity 2 query.

    Relationship contain following information:
    - Name of the entities that are related
    - In which books they were mentioned
    - How are relationships described in the books
    - What type of relationships entities share

    :param entity1_query: part of the name or type to find similar entity 1
    :param entity2_query: part of the name or type to find similar entity 2
    :return: list of relationship matching the queries
    :raises ValueError: If the input parameter entity 1 and entity 2 query is empty
    """
    if not entity1_query or not entity2_query:
        raise ValueError(
            f"You must provide a `entity1_query` and `entity2_query` to `{get_relationships_by_query.__name__}` tool!")

    rows = utils.execute_cypher(
        f"""
        MATCH (e1:Entity)-[r]-(d:Description)-[]-(e2:Entity), (d:Description)-[]-(b:Book)
        WHERE 
        (e1.name =~ $entity1_pattern OR ANY(type IN e1.entity_types WHERE type =~ $entity1_pattern))
        AND 
        (e2.name =~ $entity2_pattern OR ANY(type IN e2.entity_types WHERE type =~ $entity2_pattern))
        RETURN e1, e2, b, r, d
        """,
        entity1_pattern=f"(?i)(|.*.){".*.".join(entity1_query.split())}(|.*.)",
        entity2_pattern=f"(?i)(|.*.){".*.".join(entity2_query.split())}(|.*.)",
    )
    if not rows:
        return f"No relationships found similar to the entity1 query '{entity1_query}' and entity2 query '{entity2_query}'!"

    # Creating a nested dictionary for rows to club information together
    nested_dict = {}
    for row in rows:
        entity1_node: Node = row[0]
        entity2_node: Node = row[1]
        book_node: Node = row[2]
        relationship: Relationship = row[3]
        description_node: Node = row[4]
        description_content = description_node.get("content")
        entities_key = f"{entity1_node.id}->{entity2_node.id}"

        if entities_key not in nested_dict:
            nested_dict[entities_key] = {
                "entity1": entity1_node,
                "entity2": entity2_node,
                "books": {
                    book_node.id: {
                        "book": book_node,
                        "relationships": {relationship.type},
                        "descriptions": {description_content} if description_content else {},
                    }
                }
            }
        elif book_node.id not in nested_dict[entities_key]["books"]:
            nested_dict[entities_key]["books"][book_node.id] = {
                "book": book_node,
                "relationships": {relationship.type},
                "descriptions": {description_content} if description_content else {},
            }
        elif description_content:
            nested_dict[entities_key]["books"][book_node.id]["relationships"].add(relationship.type)
            nested_dict[entities_key]["books"][book_node.id]["descriptions"].add(description_content)

    logger.info(f"Found {len(nested_dict)} relationships | entity1: '{entity1_query}' | entity2: '{entity2_query}'")

    infos = []
    for ri, relationship_dict_value in enumerate(nested_dict.values()):
        logger.info(f"Creating info for relationship: {ri + 1}")
        entity1_node = relationship_dict_value["entity1"]
        entity2_node = relationship_dict_value["entity2"]

        # Creating top-level relationship info
        entity1_info = f"{entity1_node.get("name", "unknown")} ({"singular" if entity1_node.get("singular", True) else "plural"})"
        entity2_info = f"{entity2_node.get("name", "unknown")} ({"singular" if entity2_node.get("singular", True) else "plural"})"
        info = [f"{ri + 1}. Relationship between {entity1_info} & {entity2_info}", ""]

        # Iterating over books in which the entity is mentioned
        for book_dict_value in relationship_dict_value["books"].values():
            shortened_descriptions = utils.recursive_shortening(list(book_dict_value["descriptions"]))
            relationships = list(book_dict_value["relationships"])

            # Creating child-level book info and adding context
            book_node: Node = book_dict_value["book"]
            info += [f"{"\t" * 1}Mentioned in the book '{book_node.get("name", "unknown")}' as:", ""]
            info += [f"{"\t" * 2}- {shortened_description}" for shortened_description in shortened_descriptions]
            info += [f"{"\t" * 1}Relationship type: {", ".join(relationships)}", ""]

        infos.append("\n".join(info))

    return f"""Total: {len(nested_dict)} relationships
{"\n".join(infos)}    
    """


def build_agent(llm: ChatOllama) -> CompiledStateGraph:
    return create_react_agent(llm, tools=[
        get_books_by_query,
        get_entities_by_query,
        get_relationships_by_query,
    ], prompt=prompts.AGENT_SYSTEM_MESSAGE)
