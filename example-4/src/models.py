import logging
from typing import Any, List, Literal, Tuple, TypedDict
from uuid import uuid4

from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExtractableEntity(BaseModel):
    name: str = Field(description='Name of the entity (lowercase)')
    entity_type: str = Field(description='Type of the entity (lowercase and snake case)')
    singular: bool = Field(description='Whether the entity is singular (true) or plural (false)')
    description: str = Field(description='Description of how entity is described or introduced in the text')


class ExtractableRelationship(BaseModel):
    entity_1: str = Field(description='Name of entity 1 (lowercase)')
    entity_2: str = Field(description='Name of entity 2 (lowercase)')
    relationship_type: str = Field(description='Type of relationship between entity 1 and 2 (lowercase and snake case)')
    description: str = Field(description='Description of how relationship is described or introduced in the text')


class ExtractableEntitiesRelationships(BaseModel):
    entities: List[ExtractableEntity] = Field(description='List of entities', default=[])
    relationships: List[ExtractableRelationship] = Field(description='List of relationships between entities', default=[])


class APOCNode(BaseModel):
    type: Literal['node'] = Field(default='node')
    id: str
    labels: List[str]
    properties: dict[str, Any]


class APOCRelationship(BaseModel):
    class Node(BaseModel):
        id: str
        labels: List[str]

    type: Literal['relationship'] = Field(default='relationship')
    id: str
    label: str
    properties: dict[str, Any]
    start: Node
    end: Node


class Book(BaseModel):
    id: str
    name: str
    url: str
    start_page: int
    end_page: int

    def to_apoc_node(self) -> APOCNode:
        return APOCNode(**{
            'id': self.id,
            'labels': [Book.__name__],
            'properties': self.model_dump(),
        })


class Description(BaseModel):
    id: str
    book: Book
    chunk_index: int
    chunk_size: int
    chunk_overlap: int
    content: str

    def to_apoc_node(self) -> APOCNode:
        node_properties = self.model_dump()
        del node_properties['book']
        return APOCNode(**{
            'id': self.id,
            'labels': [Description.__name__],
            'properties': node_properties,
        })
    
    def to_apoc_relationships(self) -> List[APOCRelationship]:
        return [
            APOCRelationship(**{
                'id': str(uuid4()),
                'label': 'mentioned_in',
                'properties': {
                    'system': True,
                },
                'start': {
                    'id': self.id,
                    'labels': [Description.__name__],
                },
                'end': {
                    'id': self.book.id,
                    'labels': [Book.__name__],
                },
            }),
        ]


class Entity(BaseModel):
    id: str
    name: str
    entity_types: set[str]
    singular: bool
    descriptions: List[Description]

    @property
    def key(self) -> str:
        return f'{self.name}-{self.singular}'.lower()

    def merge(self, other: 'Entity'):
        if self.key != other.key:
            raise ValueError(f'Entity Keys should match to merge!')
        logger.debug(f'[MERGE] {Entity.__name__}({self.key})')
        self.descriptions += other.descriptions
        self.entity_types = self.entity_types.union(other.entity_types)

    def to_apoc_node(self) -> APOCNode:
        node_properties = self.model_dump()
        del node_properties['descriptions']
        return APOCNode(**{
            'id': self.id,
            'labels': [Entity.__name__],
            'properties': node_properties,
        })
    
    def to_apoc_relationships(self) -> List[APOCRelationship]:
        relationships = []
        for description in self.descriptions:
            relationships.append(APOCRelationship(**{
                'id': str(uuid4()),
                'label': 'described_as',
                'properties': {
                    'system': True,
                },
                'start': {
                    'id': self.id,
                    'labels': [Entity.__name__],
                },
                'end': {
                    'id': description.id,
                    'labels': [Description.__name__],
                },
            }))
        return relationships

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
            'entity_types': set([extractable_entity.entity_type.lower()]),
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
    
    def to_apoc_relationships(self) -> List[APOCRelationship]:
        relationships = []
        for description in self.descriptions:
            relationships += [
                # E1 -> D
                APOCRelationship(**{
                    'id': str(uuid4()),
                    'label': self.relationship_type,
                    'properties': {
                        'system': False,
                    },
                    'start': {
                        'id': self.entity_1.id,
                        'labels': [Entity.__name__],
                    },
                    'end': {
                        'id': description.id,
                        'labels': [Description.__name__],
                    }
                }),

                # D -> E2
                APOCRelationship(**{
                    'id': str(uuid4()),
                    'label': self.relationship_type,
                    'properties': {
                        'system': False,
                    },
                    'start': {
                        'id': description.id,
                        'labels': [Description.__name__],
                    },
                    'end': {
                        'id': self.entity_2.id,
                        'labels': [Entity.__name__],
                    }
                }),
            ]
        return relationships

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
        relationship_type = '_'.join(relationship_type.split('\''))
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


class BuildDatabaseAgentState(TypedDict):
    model: str
    books_json_path: str
    export_dir: str
    chunk_size: int
    chunk_overlap: int
    max_workers: int
    max_retries: int
    neo4j_uri: str
    neo4j_auth: Tuple[str, str]
    neo4j_import_dir: str

    books: List[Book]
    book_documents: dict[str, List[Document]]
    book_chunks: dict[str, List[Document]]
    book_extracts: dict[str, dict[int, ExtractableEntitiesRelationships]]
    retries: int
    entities: List[Entity]
    relationships: List[Relationship]


class ExportDirectoryMetdata(BaseModel):
    class Book(BaseModel):
        name: str
        url: str
        start_page: int
        end_page: int
        chunks: int

    model: str
    chunk_size: int
    chunk_overlap: int
    books: dict[str, Book]

    @staticmethod
    def from_state(state: BuildDatabaseAgentState) -> 'ExportDirectoryMetdata':
        books: dict[str, ExportDirectoryMetdata.Book] = dict()
        for book in state['books']:
            books[book.id] = {
                'name': book.name,
                'url': book.url,
                'start_page': book.start_page,
                'end_page': book.end_page,
                'chunks': len(state['book_chunks'][book.id]),
            }

        return ExportDirectoryMetdata(**{
            'model': state['model'],
            'chunk_size': state['chunk_size'],
            'chunk_overlap': state['chunk_overlap'],
            'books': books,
        })
