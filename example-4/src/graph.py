import json
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Tuple, TypedDict

from langchain.output_parsers.fix import OutputFixingParser
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from neo4j import GraphDatabase

from src import prompts
from src.models import Book, Entity, ExtractableEntitiesRelationships, Relationship
from src.utils import loggraph

logger = logging.getLogger(__name__)


class BuildDatabaseAgentState(TypedDict):
    llm: ChatOllama
    json_path: str
    export_dir: str
    chunk_size: int
    chunk_overlap: int
    max_workers: int
    neo4j_uri: str
    neo4j_auth: Tuple[str, str]

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
        llm = state['llm']
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

        # Exporting extracted entities & relationships
        json_file_path = os.path.join(book_dir, f'chunk-{i}.json')
        with open(json_file_path, 'w') as file:
            file.write(entities_relationships.model_dump_json(indent=2))

        logger.info(f'[{len(book_extracts[book_id])}/{len(chunks)}] Exported chunk: {i}')

    for book_id, chunks in state['chunks'].items():
        logger.info(f'Extracting entities & relationships from book: [{book_id}]')
        book_extracts[book_id] = []

        # Checking if book folder exists
        book_dir = os.path.join(state['export_dir'], book_id)
        if not os.path.exists(book_dir) or not os.path.isdir(book_dir):
            os.mkdir(book_dir)

        # Invoking LLM in multiple threads
        max_workers = min(state['max_workers'], len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, chunk in enumerate(chunks):

                # Checking if chunk already exported
                json_file_path = os.path.join(book_dir, f'chunk-{i}.json')
                if os.path.exists(json_file_path) and os.path.isfile(json_file_path):
                    with open(json_file_path, 'r') as file:
                        entities_relationships = ExtractableEntitiesRelationships.model_validate_json(file.read())
                        book_extracts[book_id].append((i, entities_relationships))
                        
                    logger.info(f'[{len(book_extracts[book_id])}/{len(chunks)}] Imported chunk: {i}')
                    continue

                # Extracting using LLM if not previously exported
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

    entities = [entity for entity in entities_dict.values()]
    relationships = [relationship for relationship in relationships_dict.values()]
    return {'entities': entities, 'relationships': relationships}


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

    conn = GraphDatabase.driver(uri=state['neo4j_uri'], auth=state['neo4j_auth'])
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
    graph.add_node('load_books', load_books)
    graph.add_node('chunk_books', chunk_books)
    graph.add_node('extract_entities_relationships', extract_entities_relationships)
    graph.add_node('process_extracted_entities_relationships', process_extracted_entities_relationships)
    graph.add_node('post_cyphers', post_cyphers)

    # Adding edges
    graph.add_edge(START, 'load_books')
    graph.add_edge('load_books', 'chunk_books')
    graph.add_edge('chunk_books', 'extract_entities_relationships')
    graph.add_edge('extract_entities_relationships', 'process_extracted_entities_relationships'),
    graph.add_edge('process_extracted_entities_relationships', 'post_cyphers')
    graph.add_edge('post_cyphers', END)

    return graph.compile()
