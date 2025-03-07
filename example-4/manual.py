from langchain_ollama import ChatOllama
from langchain.output_parsers.fix import OutputFixingParser
from langchain.output_parsers.pydantic import PydanticOutputParser

from src import prompts
from src.models import ExtractableEntitiesRelationships


i, chunk_page_content = -1, """
"""

if i < 0 or not chunk_page_content:
    raise ValueError('Set chunk index & page content!')

llm = ChatOllama(model='llama3.1:8b', temperature=0, format='json')
parser = OutputFixingParser.from_llm(
    llm=llm,
    parser=PydanticOutputParser(pydantic_object=ExtractableEntitiesRelationships),
    max_retries=3,
)
chain = prompts.EXTRACT_ENTITIES_RELATIONSHIPS_PROMPT | llm | parser
entities_relationships: ExtractableEntitiesRelationships = chain.invoke({
    'format_instructions': parser.get_format_instructions(), 
    'chunk': chunk_page_content,
})

print(entities_relationships)
with open(f'./chunk-{i}.json', 'w') as file:
    file.write(entities_relationships.model_dump_json(indent=2))