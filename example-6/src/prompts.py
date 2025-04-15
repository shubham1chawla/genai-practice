from langchain_core.prompts import ChatPromptTemplate

AGENT_SYSTEM_MESSAGE = f"""
You are an helpful assistant answering user's question.

Instructions:
1. To help you answer the question, you will be provided references from books.
2. References may include, name of the books, entities, relationships, and how they are described in the books.
3. References are given to you internally from a database and must not be mentioned to the user directly.
4. Use only the knowledge from the references to answer the question.
"""

RECURSIVE_SHORTENING_PROMPT = ChatPromptTemplate([
    ("system", """
You are an helpful assistant and your task is to shorten the user's text in one sentence.
As an assistant:
- Your shorten text should try to preserve most of the context.
- Reply in plain unformatted sentences.
- Only mention information in the text provided to you.
- Do NOT add headings, lists, or summary to the output.
    """),
    ("user", "{descriptions}")
])
