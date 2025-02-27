#!../.venv/bin/python


import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, List, Tuple, TypedDict
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


class Book(BaseModel):
    id: str
    name: str
    url: str
    start_page: int
    end_page: int

    def to_cypher(self) -> str:
        return f"""
        CREATE (:{Book.__name__} {{
            id: "{self.id}",
            name: "{self.name}",
            url: "{self.url}",
            start_page: {self.start_page},
            end_page: {self.end_page}
        }})
        """


class Description(BaseModel):
    id: str
    book: Book
    chunk_index: int
    chunk_size: int
    chunk_overlap: int
    content: str

    def to_cypher(self) -> str:
        return f"""
        // CREATES DESCRIPTION NODE & LINKS WITH BOOK
        CREATE (d:{Description.__name__} {{
            id: "{self.id}",
            content: "{self.content}",
            chunk_index: {self.chunk_index},
            chunk_size: {self.chunk_size},
            chunk_overlap: {self.chunk_overlap}
        }})
        WITH d
        MATCH (b:{Book.__name__} {{ id: "{self.book.id}" }})
        CREATE (d)-[:mentioned_in {{ system: true }}]->(b)
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
        logger.debug(f'[MERGE] {Entity.__name__}({self.key})')
        self.descriptions += other.descriptions

    def to_cypher(self) -> str:
        match_cyphers, create_cyphers = [], []
        for i, description in enumerate(self.descriptions):
            match_cyphers.append(f'(d_{i}:{Description.__name__} {{ id: "{description.id}" }})')
            create_cyphers.append(f'CREATE (e)-[:described_as {{ system: true }}]->(d_{i})')

        return f"""
        // CREATES ENTITY NODE & LINKS WITH DESCRIPTIONS
        CREATE (e:{Entity.__name__} {{
            id: "{self.id}", 
            name: "{self.name}", 
            entity_name: "{self.entity_type}",
            singular: {json.dumps(self.singular)}
        }})
        WITH e
        MATCH {', '.join(match_cyphers)}
        {'\n'.join(create_cyphers)}
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
                'id': str(uuid4()),
                'book': book,
                'chunk_index': chunk_index,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'content': extractable_entity.description,
            })],
        })
    
    @staticmethod
    def from_path(path: str) -> List['Entity']:
        if not os.path.exists(path):
            raise ValueError(f'{path} does not exists!')
        
        with open(path, 'r') as file:
            return [Entity.model_validate(e) for e in json.loads(file.read())]


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
        logger.debug(f'[MERGE] {Relationship.__name__}({self.key})')
        self.descriptions += other.descriptions

    def to_cypher(self) -> str:
        create_cyphers, match_cyphers = [], [
            f'(e1:{Entity.__name__} {{id: "{self.entity_1.id}"}})',
            f'(e2:{Entity.__name__} {{id: "{self.entity_2.id}"}})',
        ]
        for i, description in enumerate(self.descriptions):
            match_cyphers.append(f'(d_{i}:{Description.__name__} {{ id: "{description.id}" }})')
            create_cyphers.append(f"""
            CREATE (e1)-[:{self.relationship_type} {{ system: false }}]->(d_{i})-[:{self.relationship_type} {{ system: false }}]->(e2)
            """)

        return f"""
        // FINDS ENTITIES & DESCRIPTIONS
        MATCH {', '.join(match_cyphers)}

        // CREATING LINKS BETWEEN ENTITY 1 & 2
        {'\n'.join(create_cyphers)}
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
        relationship_type = '_'.join(relationship_type.split())
        relationship_type = '_'.join(relationship_type.split('-'))
        return Relationship(**{
            'id': str(uuid4()),
            'entity_1': entities[0],
            'entity_2': entities[1],
            'relationship_type': relationship_type.lower(),
            'descriptions': [Description(**{
                'id': str(uuid4()),
                'book': book,
                'chunk_index': chunk_index,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'content': description,
            })],
        })
    
    @staticmethod
    def from_path(path: str) -> List['Relationship']:
        if not os.path.exists(path):
            raise ValueError(f'{path} does not exists!')

        with open(path, 'r') as file:
            return [Relationship.model_validate(r) for r in json.loads(file.read())]


class BuildDatabaseAgentState(TypedDict):
    json_path: str
    export_dir: str
    chunk_size: int
    chunk_overlap: int
    max_workers: int

    books: List[Book]
    book_documents: dict[str, List[Document]]
    chunks: dict[str, List[Document]]
    book_extracts: dict[str, List[Tuple[int, ExtractableEntitiesRelationships]]]
    entities: List[Entity]
    relationships: List[Relationship]


@loggraph
def load_books(state: BuildDatabaseAgentState):
    # Loading books info from JSON
    with open(state['json_path'], 'r') as file:
        dicts: List[dict[str, Any]] = json.loads(file.read())
    
    # Loading PDF files
    books: List[Book] = []
    book_documents: dict[str, List[Document]] = dict()
    for d in dicts:
        loader = PyPDFLoader(file_path=d['url'], headers={'User-Agent': 'Mozilla/5.0'})
        documents = loader.load()
        start_page, end_page = 1, len(documents)

        if d.get('start_page', None) is not None and d['start_page'] > 1:
            start_page = d['start_page']
        else:
            d['start_page'] = start_page
        
        if d.get('end_page', None) is not None and d['end_page'] < len(documents):
            end_page = d['end_page']
        else:
            d['end_page'] = end_page

        logger.info(f'Loaded Book[id: {d['id']}, pages: {len(documents)}] | Indexes: [{start_page-1}, {end_page})')
        documents = documents[start_page-1:end_page]

        book = Book.model_construct(**d)
        books.append(book)
        book_documents[book.id] = documents

    return {'books': books, 'book_documents': book_documents}


@loggraph
def resume_process(state: BuildDatabaseAgentState):
    entities_file_path = os.path.join(state['export_dir'], 'entities.json')
    relationship_file_path = os.path.join(state['export_dir'], 'relationships.json')
    
    # If files don't exist
    if not os.path.exists(entities_file_path) or not os.path.exists(relationship_file_path):
        logger.info(f'Entities or relationships not exported, restarting...')
        return 'chunk_books'

    return 'load_entities_relationships'


@loggraph
def load_entities_relationships(state: BuildDatabaseAgentState):
    entities_file_path = os.path.join(state['export_dir'], 'entities.json')
    relationship_file_path = os.path.join(state['export_dir'], 'relationships.json')

    # Loading entities
    entities = Entity.from_path(entities_file_path)
    logger.info(f'Loaded {len(entities)} entities from {entities_file_path}')

    # Loading relationships
    relationships = Relationship.from_path(relationship_file_path)
    logger.info(f'Loaded {len(relationships)} relationships from {entities_file_path}')

    return {'entities': entities, 'relationships': relationships}


@loggraph
def chunk_books(state: BuildDatabaseAgentState):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=state['chunk_size'], 
        chunk_overlap=state['chunk_overlap'],
    )

    chunks: dict[str, List[Document]] = dict()
    for book_id, documents in state['book_documents'].items():
        chunks[book_id] = text_splitter.split_documents(documents)
        logger.info(f'Chunked book [{book_id}] into {len(chunks[book_id])} documents')
    
    return {'chunks': chunks}


@loggraph
def extract_entities_relationships(state: BuildDatabaseAgentState):
    book_extracts: dict[str, List[Tuple[int, ExtractableEntitiesRelationships]]] = dict()

    def extract_from_chunk(i: int, chunk: Document) -> Tuple[int, ExtractableEntitiesRelationships]:
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
        return i, entities_relationships

    def callback(book_id: str, future: Future[Tuple[int, ExtractableEntitiesRelationships]]):
        i, entities_relationships = future.result()
        book_extracts[book_id].append((i, entities_relationships))
        logger.info(f'[{len(book_extracts[book_id])}/{len(chunks)}] Completed chunk: {i}')

    for book_id, chunks in state['chunks'].items():
        logger.info(f'Extracting entities & relationships from book: [{book_id}]')
        book_extracts[book_id] = []

        # Invoking LLM in multiple threads
        max_workers = min(state['max_workers'], len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, chunk in enumerate(chunks):
                future = executor.submit(extract_from_chunk, i, chunk)
                future.add_done_callback(lambda f: callback(book_id, f))

    return {'book_extracts': book_extracts}


@loggraph
def process_extracted_entities_relationships(state: BuildDatabaseAgentState):
    # Creating books dict
    book_dict = {book.id: book for book in state['books']}

    # Entity Key -> Entity
    entities_dict: dict[str, Entity] = dict()
    
    # Relationship Key -> Relationship
    relationships_dict: dict[str, Relationship] = dict()

    for book_id, extracts in state['book_extracts'].items():
        logger.info(f'Processing entities & relationships from book: [{book_id}]')
        book = book_dict[book_id]

        for i, entities_relationships in extracts:
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

            logger.info(f'[TE: {len(entities_dict)}, TR: {len(relationships_dict)}] Completed chunk: {i}')

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
    entities_json_path = os.path.join(export_dir, 'entities.json')
    with open(entities_json_path, 'w') as file:
        file.write(json.dumps([entity.model_dump() for entity in entities], indent=2))
    logger.info(f'Exported {len(entities)} entities to "{entities_json_path}"')

    # Exporting relationships
    relationships_json_path = os.path.join(export_dir, 'relationships.json')
    with open(relationships_json_path, 'w') as file:
        file.write(json.dumps([relationship.model_dump() for relationship in relationships], indent=2))
    logger.info(f'Exported {len(relationships)} relationships to "{relationships_json_path}"')


@loggraph
def post_cyphers(state: BuildDatabaseAgentState):
    cyphers = [f'MATCH (n) DETACH DELETE n']
    cyphers += [book.to_cypher() for book in state['books']]

    # Adding entities & descriptions
    for entity in state['entities']:
        cyphers += [description.to_cypher() for description in entity.descriptions]
        cyphers.append(entity.to_cypher())

    # Adding relationships & descriptions
    for relationship in state['relationships']:
        cyphers += [description.to_cypher() for description in relationship.descriptions]
        cyphers.append(relationship.to_cypher())

    conn = GraphDatabase.driver(uri=constants.NEO4J_URI, auth=constants.NEO4J_AUTH)
    session = conn.session()
    tx = session.begin_transaction()
    for cypher in cyphers:
        tx.run(cypher)
    tx.commit()

    filename = os.path.join(state['export_dir'], 'cyphers.json')
    with open(filename, 'w') as file:
        file.write(json.dumps(cyphers, indent=2))
    logger.info(f'Posted & exported cyphers to {filename}')

    conn.close()


def build_graph() -> CompiledGraph:
    logger.info(f'Building {BuildDatabaseAgentState.__name__} graph...')
    graph = StateGraph(BuildDatabaseAgentState)

    # Adding nodes
    graph.add_node('load_entities_relationships', load_entities_relationships)
    graph.add_node('load_books', load_books)
    graph.add_node('chunk_books', chunk_books)
    graph.add_node('extract_entities_relationships', extract_entities_relationships)
    graph.add_node('process_extracted_entities_relationships', process_extracted_entities_relationships)
    graph.add_node('export_entities_relationships', export_entities_relationships)
    graph.add_node('post_cyphers', post_cyphers)

    # Adding conditional edges
    graph.add_conditional_edges(
        'load_books', resume_process,
        {
            'chunk_books': 'chunk_books',
            'load_entities_relationships': 'load_entities_relationships',
        }
    )

    # Adding edges
    graph.add_edge(START, 'load_books')
    graph.add_edge('load_entities_relationships', 'post_cyphers')
    graph.add_edge('chunk_books', 'extract_entities_relationships')
    graph.add_edge('extract_entities_relationships', 'process_extracted_entities_relationships'),
    graph.add_edge('process_extracted_entities_relationships', 'export_entities_relationships')
    graph.add_edge('export_entities_relationships', 'post_cyphers')
    graph.add_edge('post_cyphers', END)

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
        'max_workers': constants.MAX_WORKERS,
    })
