import json
import logging
from typing import List, Tuple
from uuid import uuid4

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
            content: {json.dumps(self.content)},
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
