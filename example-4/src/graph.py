import json
import logging
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Tuple

from langchain.output_parsers.fix import OutputFixingParser
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from neo4j import GraphDatabase, ManagedTransaction, ResultSummary

from src import prompts
from src.models import (
    APOCNode, APOCRelationship, Book, BuildDatabaseAgentState, Description, Entity, 
    ExportDirectoryMetdata, ExtractableEntitiesRelationships, Relationship
)
from src.utils import loggraph, validate_export_metadata

logger = logging.getLogger(__name__)


@loggraph
def load_books(state: BuildDatabaseAgentState):
    # Loading books info from JSON
    with open(state['books_json_path'], 'r') as file:
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

    book_chunks: dict[str, List[Document]] = dict()
    for book_id, documents in state['book_documents'].items():
        book_chunks[book_id] = text_splitter.split_documents(documents)
        logger.info(f'[{book_id}] Chunked book into {len(book_chunks[book_id])} documents')
    
    return {'book_chunks': book_chunks}


@loggraph
def validate_metadata(state: BuildDatabaseAgentState):
    try:
        validate_export_metadata(state)
        return 'export_metadata'
    except Exception as e:
        logger.error(e)
        return END


@loggraph
def export_metadata(state: BuildDatabaseAgentState):

    # Creating export directory if doesn't exists
    export_dir = state['export_dir']
    if not os.path.exists(export_dir) or not os.path.isdir(export_dir):
        logger.info(f'Creating export directory: "{export_dir}"')
        os.mkdir(export_dir)

    # Exporting metadata
    metadata = ExportDirectoryMetdata.from_state(state)
    metadata_json_path = os.path.join(export_dir, 'metadata.json')
    with open(metadata_json_path, 'w') as file:
        file.write(metadata.model_dump_json(indent=2))

    logger.info(f'Exported metadata: "{metadata_json_path}"')


@loggraph
def extract_entities_relationships(state: BuildDatabaseAgentState):
    book_extracts: dict[str, dict[int, ExtractableEntitiesRelationships]] = dict()

    # Loading already exported entities & relationships
    for book_id, chunks in state['book_chunks'].items():
        book_extracts[book_id] = dict()

        # Checking if book folder exists
        book_dir = os.path.join(state['export_dir'], book_id)
        if not os.path.exists(book_dir) or not os.path.isdir(book_dir):
            continue

        chunk_jsons = [filename for filename in os.listdir(book_dir) if filename.startswith('chunk-') and filename.endswith('.json')]
        if not chunk_jsons:
            continue

        # Reading exported files from book directory
        for filename in chunk_jsons:
            i = int(re.findall(r'\d+', filename)[0])
            
            file_path = os.path.join(book_dir, filename)
            with open(file_path, 'r') as file:
                book_extracts[book_id][i] = ExtractableEntitiesRelationships.model_validate_json(file.read())
            
            logger.debug(f'[{book_id}] [{len(book_extracts[book_id])}/{len(chunks)}] Imported chunk: {i}')
        
        logger.info(f'[{book_id}] Imported entities & relationships from {len(book_extracts[book_id])}/{len(chunks)} chunks')

    def extract_from_chunk(i: int, chunk: Document) -> Tuple[int, ExtractableEntitiesRelationships]:
        try:
            llm = ChatOllama(model=state['model'], temperature=0, format='json')
            parser = OutputFixingParser.from_llm(
                llm=llm,
                parser=PydanticOutputParser(pydantic_object=ExtractableEntitiesRelationships),
                max_retries=state['max_retries'],
            )
            chain = prompts.EXTRACT_ENTITIES_RELATIONSHIPS_PROMPT | llm | parser
            entities_relationships: ExtractableEntitiesRelationships = chain.invoke({
                'format_instructions': parser.get_format_instructions(), 
                'chunk': chunk.page_content,
            })
            return i, entities_relationships
        except Exception as e:
            logger.error(f'[{book_id}] [CHUNK-{i}] Unable to extract entities & relationships! Chunk: {chunk}')
            logger.debug(e)

    def callback(book_id: str, future: Future[Tuple[int, ExtractableEntitiesRelationships]]):
        if not future.result():
            return

        i, entities_relationships = future.result()
        book_extracts[book_id][i] = entities_relationships

        # Checking if book folder exists
        book_dir = os.path.join(state['export_dir'], book_id)
        if not os.path.exists(book_dir) or not os.path.isdir(book_dir):
            logger.info(f'[{book_id}] Creating directory')
            os.mkdir(book_dir)

        # Exporting extracted entities & relationships
        json_file_path = os.path.join(book_dir, f'chunk-{i}.json')
        with open(json_file_path, 'w') as file:
            file.write(entities_relationships.model_dump_json(indent=2))

        logger.info(f'[{book_id}] [{len(book_extracts[book_id])}/{len(chunks)}] Exported chunk: {i}')

    for book_id, chunks in state['book_chunks'].items():
    
        # Filtering imported chunks
        filtered_chunks = list(filter(lambda tuple: tuple[0] not in book_extracts[book_id], enumerate(chunks)))
        if not filtered_chunks:
            logger.info(f'[{book_id}] All entities & relationships imported')
            continue

        logger.info(f'[{book_id}] Extracting entities & relationships from {len(filtered_chunks)}/{len(chunks)} chunks')

        # Invoking LLM in multiple threads
        max_workers = min(len(filtered_chunks), state['max_workers'])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            # Extracting using LLM if not previously exported
            for i, chunk in filtered_chunks:
                future = executor.submit(extract_from_chunk, i, chunk)
                future.add_done_callback(lambda f: callback(book_id, f))

    retries = state.get('retries', 0) + 1
    return {'book_extracts': book_extracts, 'retries': retries}


@loggraph
def check_extracted_entities_relationships(state: BuildDatabaseAgentState):
    # Checking if max retries reached
    if state['retries'] == state['max_retries']:
        logger.error(f'Max retries reached, unable to build the database!')
        return END

    # Checking if all entities & relationships are imported/exported
    for book_id, chunks in state['book_chunks'].items():
        if len(chunks) != len(state['book_extracts'][book_id]):
            logger.warning(f'[{book_id}] Detected missing chunks, retrying extraction process...')
            return 'extract_entities_relationships'
        
        logger.info(f'[{book_id}] All entities & relationship imported or exported!')

    return 'process_extracted_entities_relationships'


@loggraph
def process_extracted_entities_relationships(state: BuildDatabaseAgentState):
    # Creating books dict
    book_dict = {book.id: book for book in state['books']}

    # Entity Key -> Entity
    entities_dict: dict[str, Entity] = dict()
    
    # Relationship Key -> Relationship
    relationships_dict: dict[str, Relationship] = dict()

    for book_id, extracts in state['book_extracts'].items():
        logger.info(f'[{book_id}] Processing entities & relationships from {len(extracts)} chunks')
        book = book_dict[book_id]

        for i, entities_relationships in extracts.items():
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
                    logger.debug(f'Relationship\'s Entity "{entity_1_name if not entity_1 else entity_2_name}" is missing!')

            logger.debug(f'[{book_id}] [{i+1}/{len(extracts)}] TE: {len(entities_dict)} TR: {len(relationships_dict)}')

        logger.info(f'[{book_id}] TE: {len(entities_dict)} TR: {len(relationships_dict)}')

    entities = [entity for entity in entities_dict.values()]
    relationships = [relationship for relationship in relationships_dict.values()]
    return {'entities': entities, 'relationships': relationships}


@loggraph
def post_cyphers(state: BuildDatabaseAgentState):
    # Checking neo4j import directory
    if not os.path.exists(state['neo4j_import_dir']) or not os.path.isdir(state['neo4j_import_dir']):
        raise ValueError(f'Neo4j import directory not found! Did you forgot to run "docker compose up -d"?')

    apoc_nodes: List[APOCNode] = []
    apoc_relationships: List[APOCRelationship] = []

    # Adding books
    for book in state['books']:
        apoc_nodes.append(book.to_apoc_node())

    # Adding entities
    for entity in state['entities']:
        for description in entity.descriptions:
            apoc_nodes.append(description.to_apoc_node())
            apoc_relationships += description.to_apoc_relationships()
        apoc_nodes.append(entity.to_apoc_node())
        apoc_relationships += entity.to_apoc_relationships()
    
    # Adding relationships
    for relationship in state['relationships']:
        for description in relationship.descriptions:
            apoc_nodes.append(description.to_apoc_node())
            apoc_relationships += description.to_apoc_relationships()
        apoc_relationships += relationship.to_apoc_relationships()

    # Exporting APOC JSON files
    apoc_jsons = '\n'.join([apoc_entity.model_dump_json() for apoc_entity in apoc_nodes + apoc_relationships])
    filename = 'apoc.json'
    file_paths = [
        os.path.join(state['neo4j_import_dir'], filename),
        os.path.join(state['export_dir'], filename),
    ]
    for file_path in file_paths:
        with open(file_path, 'w') as file:
            file.write(apoc_jsons)

    driver = GraphDatabase.driver(uri=state['neo4j_uri'], auth=state['neo4j_auth'])
    with driver.session() as session:

        def executor(tx: ManagedTransaction, query: str, params):
            result = tx.run(query, **params)
            summary: ResultSummary = result.consume()
            logger.info(f'Executed: {query} | {summary.counters}')

        queries = [
            ('MATCH (n) DETACH DELETE n', {}), # Deleting old entities & relationships
            ('CALL apoc.schema.assert({}, {})', {}), # Removing all constaints
        ]

        # Adding new constraints queries
        for label in [Book.__name__, Description.__name__, Entity.__name__]:
            queries.append((f'CREATE CONSTRAINT FOR (n:{label}) REQUIRE n.neo4jImportId IS UNIQUE;', {}))
        
        # Adding final APOC json import query
        queries.append((
            'CALL apoc.import.json($file_path)',
            {'file_path': f'file:///var/lib/neo4j/import/{filename}'}
        ))

        # Executing queries
        try:
            for query, params in queries:
                session.execute_write(executor, query, params)
        except Exception as e:
            logger.error(f'Unable to create database!')
            logger.debug(e)

    driver.close()


def build_graph() -> CompiledGraph:
    logger.info(f'Building {BuildDatabaseAgentState.__name__} graph...')
    graph = StateGraph(BuildDatabaseAgentState)

    # Adding nodes
    graph.add_node('load_books', load_books)
    graph.add_node('chunk_books', chunk_books)
    graph.add_node('export_metadata', export_metadata)
    graph.add_node('extract_entities_relationships', extract_entities_relationships)
    graph.add_node('process_extracted_entities_relationships', process_extracted_entities_relationships)
    graph.add_node('post_cyphers', post_cyphers)

    # Adding conditional edges
    graph.add_conditional_edges(
        'chunk_books', validate_metadata,
        {
            END: END,
            'export_metadata': 'export_metadata',
        },
    )
    graph.add_conditional_edges(
        'extract_entities_relationships', check_extracted_entities_relationships,
        {
            END: END,
            'extract_entities_relationships': 'extract_entities_relationships',
            'process_extracted_entities_relationships': 'process_extracted_entities_relationships',
        }
    )

    # Adding edges
    graph.add_edge(START, 'load_books')
    graph.add_edge('load_books', 'chunk_books')
    graph.add_edge('export_metadata', 'extract_entities_relationships')
    graph.add_edge('process_extracted_entities_relationships', 'post_cyphers')
    graph.add_edge('post_cyphers', END)

    return graph.compile()
