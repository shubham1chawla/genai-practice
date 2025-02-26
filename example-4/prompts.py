from langchain.prompts.chat import ChatPromptTemplate


EXTRACT_ENTITIES_RELATIONSHIPS_PROMPT = ChatPromptTemplate([
    ('system', """
     Your task is to extract list of entities and their relationships from the text given to you.

     Important instructions on extracting entities -
     1. Make sure that the name of the entities are proper nouns and does not contain adjectives, pronouns, and prepositions. 
     2. Only include entities which you think are important and relevant from the text.
     3. Avoid generic entities.
     4. Entity Type should be in lowercase snake_case. For instance, 'parent_child', 'involved_in'.
     5. Name of entities should be lowercase.
     
     Important instructions on extracting relationships -
     1. Only extract relationships for entities you will be mentioning.
     2. Ensure you only include relationships between entities that you have extracted.
     3. Relationship Type should be in lowercase snake_case. For instance, 'parent_child', 'involved_in'.
     4. Name of entities 1 & 2 should match with the list of entities.
     5. Entities 1 & 2 both should be part of entities for them to have a relationship.
     6. Relationship is directed and defined from Entity 1 to Entity 2, i.e. e1 -> e2 
     7. If a relationship is bidirectional, inverse reverse relationship, i.e. e2 -> e1

     Formatting instructions -
     {format_instructions}
    """),
    ('human', '{chunk}'),
])
