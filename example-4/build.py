#!../.venv/bin/python


import json
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple, TypedDict
from uuid import uuid4

from langchain.output_parsers.fix import OutputFixingParser
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

import constants
import prompts


def loggraph(func):
    def wrapper(*args, **kwargs):
        logger.info(f'{'-' * 100}')
        logger.info(f"STARTED '{func.__name__}'")
        logger.info(f'{'-' * 100}')
        
        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()

        logger.info(f'{'-' * 100}')
        logger.info(f"ENDED '{func.__name__}' IN {end_time - start_time:.2f} SECONDS")
        logger.info(f'{'-' * 100}')
        return output
    return wrapper


class ExtractableEntity(BaseModel):
    name: str = Field(description='Name of the entity (lowercase)')
    entity_type: str = Field(description='Type of the entity (lowercase and snake case)')
    singular: bool = Field(description='Whether the entity is singular (true) or plural (false)')
    description: str = Field(description='Summarize exactly how the entity is described or introduced in the text')


class ExtractableRelationship(BaseModel):
    entity_1: str = Field(description='Name of entity 1 (lowercase)')
    entity_2: str = Field(description='Name of entity 2 (lowercase)')
    relationship_type: str = Field(description='Type of relationship between entity 1 and 2 (lowercase and snake case)')
    description: str = Field(description='Summarize exactly how the relationship is described or introduced in the text')


class ExtractableEntitiesRelationships(BaseModel):
    entities: List[ExtractableEntity] = Field(description='List of entities')
    relationships: List[ExtractableRelationship] = Field(description='List of relationships between entities')


class Book(TypedDict):
    id: str
    name: str
    url: str
    start_page: int
    end_page: int
    documents: List[Document]


class Description(BaseModel):
    book_id: str
    book_name: str
    book_url: str
    book_start_page: int
    book_end_page: int
    chunk_index: int
    chunk_size: int
    chunk_overlap: int
    content: str

    def __str__(self) -> str:
        return f"""content={self.content}
book.id={self.book_id}
book.name={self.book_name}
book.url={self.book_url}
book.start_page={self.book_start_page}
book.end_page={self.book_end_page}
chunk.index={self.chunk_index}
chunk.size={self.chunk_size}
chunk.overlap={self.chunk_overlap}
"""

class Entity(BaseModel):
    id: str
    name: str
    entity_type: str
    singular: bool
    descriptions: List[Description]

    @property
    def key(self) -> str:
        return f'{self.name}-{self.entity_type}-{self.singular}'.lower()

    def merge(self, other: 'Entity'):
        if self.key != other.key:
            raise ValueError(f'Entity Keys should match to merge!')
        logger.info(f'[MERGE] {Entity.__name__}({self.key})')
        self.descriptions += other.descriptions

    def to_cypher(self) -> str:
        return f"""
        CREATE (:{Entity.__name__} {{
            id: "{self.id}", 
            name: "{self.name}", 
            entity_name: "{self.entity_type}",
            singular: {json.dumps(self.singular)},
            description: {json.dumps(list(map(str, self.descriptions)))}
        }})
        """

    @staticmethod
    def from_extractable_entity(
        extractable_entity: ExtractableEntity,
        book: Book, 
        chunk_index: int, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> 'Entity':
        return Entity(**{
            'id': str(uuid4()),
            'name': extractable_entity.name.lower(),
            'entity_type': extractable_entity.entity_type.lower(),
            'singular': extractable_entity.singular,
            'descriptions': [Description(**{
                'book_id': book['id'],
                'book_name': book['name'],
                'book_url': book['url'],
                'book_start_page': book['start_page'],
                'book_end_page': book['end_page'],
                'chunk_index': chunk_index,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'content': extractable_entity.description,
            })],
        })


class Relationship(BaseModel):
    id: str
    entity_1: Entity
    entity_2: Entity
    relationship_type: str
    descriptions: List[Description]

    @property
    def key(self) -> str:
        return f'{self.entity_1.key}>[{self.relationship_type}]>{self.entity_2.key}'

    def merge(self, other: 'Relationship'):
        if self.key != other.key:
            raise ValueError(f'Relationship Keys should match to merge!')
        logger.info(f'[MERGE] {Relationship.__name__}({self.key})')
        self.descriptions += other.descriptions

    def to_cypher(self) -> str:
        return f"""
        MATCH (e1:{Entity.__name__} {{id: "{self.entity_1.id}"}}), (e2:{Entity.__name__} {{id: "{self.entity_2.id}"}})
        CREATE (e1)-[:{self.relationship_type} {{
            id: "{self.id}",
            description: {json.dumps(list(map(str, self.descriptions)))}
        }}]->(e2)
        """

    @staticmethod
    def from_entities(
        entities: Tuple[Entity, Entity], 
        relationship_type: str, 
        description: str,
        book: Book, 
        chunk_index: int, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> 'Relationship':
        return Relationship(**{
            'id': str(uuid4()),
            'entity_1': entities[0],
            'entity_2': entities[1],
            'relationship_type': relationship_type,
            'descriptions': [Description(**{
                'book_id': book['id'],
                'book_name': book['name'],
                'book_url': book['url'],
                'book_start_page': book['start_page'],
                'book_end_page': book['end_page'],
                'chunk_index': chunk_index,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'content': description,
            })],
        })


class BuildDatabaseAgentState(TypedDict):
    json_path: str
    export_dir: str
    chunk_size: int
    chunk_overlap: int
    max_threads: int

    books: List[Book]
    chunks: dict[str, List[Document]]
    entities: List[Entity]
    relationships: List[Relationship]


@loggraph
def load_books(state: BuildDatabaseAgentState):
    # Loading books info from JSON
    with open(state['json_path'], 'r') as file:
        books: List[Book] = json.loads(file.read(), object_hook=lambda d: Book(**d))
    
    # Loading PDF files
    for book in books:
        loader = PyPDFLoader(file_path=book['url'], headers={'User-Agent': 'Mozilla/5.0'})
        book['documents'] = loader.load()[book['start_page']:book['end_page']]
        logger.info(f'Loaded books: {book['id']}')

    return {'books': books}


@loggraph
def chunk_books(state: BuildDatabaseAgentState):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=state['chunk_size'], 
        chunk_overlap=state['chunk_overlap'],
    )

    all_chunks = dict()
    for book in state['books']:
        all_chunks[book['id']] = text_splitter.split_documents(book['documents'])
        logger.info(f'Chunked book [{book['id']}] into {len(all_chunks[book['id']])} documents')
    
    return {'chunks': all_chunks}


@loggraph
def extract_entities_relationships(state: BuildDatabaseAgentState):
    # Creating books dict
    book_dict = {book['id']: book for book in state['books']}

    # Entity Key -> Entity
    entities_dict: dict[str, Entity] = dict()
    
    # Relationship Key -> Relationship
    relationships_dict: dict[str, Relationship] = dict()

    def extract_from_chunk(i: int, chunk: Document):
        parser = OutputFixingParser.from_llm(
            llm=llm,
            parser=PydanticOutputParser(pydantic_object=ExtractableEntitiesRelationships),
            max_retries=3,
        )
        chain = prompts.EXTRACT_ENTITIES_RELATIONSHIPS_PROMPT | llm | parser
        entities_relationships: ExtractableEntitiesRelationships = chain.invoke({
            'format_instructions': parser.get_format_instructions(), 
            'chunk': chunk.page_content,
        })

        """
        Creating a temp dict to store the relationship between extractable entity
        and the final entity for identifying relationships
        """
        extractable_entity_dict: dict[str, Entity] = dict()
        for extractable_entity in entities_relationships.entities:
            # Creating entity instance
            entity = Entity.from_extractable_entity(
                extractable_entity, 
                book, i, state['chunk_size'], state['chunk_overlap']
            )
            
            # Merging entities if already existing
            if entity.key in entities_dict:
                entities_dict[entity.key].merge(entity)
            else:
                entities_dict[entity.key] = entity

            # Storing temp link between extractable entity and final entity
            extractable_entity_dict[extractable_entity.name] = entities_dict[entity.key]

        for extractable_relationship in entities_relationships.relationships:
            entity_1_name, entity_2_name = extractable_relationship.entity_1, extractable_relationship.entity_2

            # Finding final entity instance from entities' names
            entity_1 = extractable_entity_dict.get(entity_1_name, None)
            entity_2 = extractable_entity_dict.get(entity_2_name, None)

            if entity_1 and entity_2:
                # Creating relationship instance
                relationship = Relationship.from_entities(
                    (entity_1, entity_2),
                    extractable_relationship.relationship_type,
                    extractable_relationship.description,
                    book, i, state['chunk_size'], state['chunk_overlap']
                )

                # Merging relationships if already existing
                if relationship.key in relationships_dict:
                    relationships_dict[relationship.key].merge(relationship)
                else:
                    relationships_dict[relationship.key] = relationship
            else:
                logger.warning(f'Relationship\'s Entity "{entity_1_name if not entity_1 else entity_2_name}" is missing!')
        return i

    for book_id, chunks in state['chunks'].items():
        logger.info(f'Extracting entities & relationships from book: [{book_id}]')
        book = book_dict[book_id]
        
        max_workers = min(state['max_threads'], len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            # Invoking LLM in multiple threads
            futures: deque[Future[Tuple[int, ExtractableEntitiesRelationships]]] = deque()
            for i, chunk in enumerate(chunks):
                futures.append(executor.submit(extract_from_chunk, i, chunk))

            # Waiting for threads to complete execution
            while futures:
                future = futures.popleft()
                if future.done():
                    i = future.result()
                    logger.info(' '.join([
                        f'[CHUNK-{i}|{len(chunks) - len(futures)}/{len(chunks)}]',
                        f'TE: {len(entities_dict)} & TR: {len(relationships_dict)}'
                    ])) 
                else:
                    futures.append(future)

    return {
        'entities': [entity for entity in entities_dict.values()], 
        'relationships': [relationship for relationship in relationships_dict.values()]
    }


@loggraph
def export_entities_relationships(state: BuildDatabaseAgentState):
    export_dir, entities, relationships = state['export_dir'], state['entities'], state['relationships']

    # Checking if export directory exists 
    if not os.path.exists(export_dir):
        logger.info(f'Creating directory: {export_dir}')
        os.mkdir(export_dir)

    # Exporting entities
    filename = os.path.join(export_dir, 'entities.json')
    with open(filename, 'w') as file:
        file.write(json.dumps([entity.model_dump() for entity in entities], indent=2))
    logger.info(f'Exported {len(entities)} entities to "{filename}"')

    # Exporting relationships
    filename = os.path.join(export_dir, 'relationships.json')
    with open(filename, 'w') as file:
        file.write(json.dumps([relationship.model_dump() for relationship in relationships], indent=2))
    logger.info(f'Exported {len(relationships)} relationships to "{filename}"')


@loggraph
def post_entities_relationships(state: BuildDatabaseAgentState):
    cyphers = [f'MATCH (e:{Entity.__name__}) DETACH DELETE e']
    cyphers += [entity.to_cypher() for entity in state['entities']]
    cyphers += [relationship.to_cypher() for relationship in state['relationships']]

    conn = GraphDatabase.driver(uri=constants.NEO4J_URI, auth=constants.NEO4J_AUTH)
    session = conn.session()
    for cypher in cyphers:
        session.run(cypher)
    logger.info(f'Posted {len(cyphers)} cyphers')

    conn.close()


def build_graph() -> CompiledGraph:
    logger.info(f'Building {BuildDatabaseAgentState.__name__} graph...')
    graph = StateGraph(BuildDatabaseAgentState)

    # Adding nodes
    graph.add_node('load_books', load_books)
    graph.add_node('chunk_books', chunk_books)
    graph.add_node('extract_entities_relationships', extract_entities_relationships)
    graph.add_node('export_entities_relationships', export_entities_relationships)
    graph.add_node('post_entities_relationships', post_entities_relationships)

    # Adding edges
    graph.add_edge(START, 'load_books')
    graph.add_edge('load_books', 'chunk_books')
    graph.add_edge('chunk_books', 'extract_entities_relationships')
    graph.add_edge('extract_entities_relationships', 'export_entities_relationships')
    graph.add_edge('export_entities_relationships', 'post_entities_relationships')
    graph.add_edge('post_entities_relationships', END)

    return graph.compile()


if __name__ == "__main__":
    # Setting up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(os.path.basename(__file__))
    for mute_packages in ['httpx', 'neo4j']:
        logging.getLogger(mute_packages).setLevel(logging.WARNING)

    llm = ChatOllama(model=constants.LLM_MODEL_NAME, temperature=0, format="json")
    build_graph().invoke({
        'json_path': constants.BOOKS_JSON_PATH,
        'export_dir': constants.EXPORT_DIR,
        'chunk_size': constants.CHUNK_SIZE,
        'chunk_overlap': constants.CHUNK_OVERLAP,
        'max_threads': constants.MAX_THREADS,
    })
