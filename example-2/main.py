#!.venv/bin/python

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import List, TypedDict

import os

MODEL_NAME = "llama3.1:8b"
COLLECTION_NAME = "example-2"
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "./db")


def create_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    sources: List[str]


def retrieve(state: State):
    all_docs = {}

    # Adding docs fetched from user's question
    for doc in vector_store.similarity_search(state["question"], k=6):
        all_docs[f"{doc.metadata["source"]}-{doc.metadata["page"]}"] = doc

    # Creating LLM model
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.3, verbose=True)

    # Rephrasing the question to find better resources
    response = llm.invoke([
        ("system", """
        You are an expert at rephrasing questions, and are tasked to elaborate user's simple 
        cooking-related question into a more specific & directed question. 
        Your rephrased question shouldn't distort user's question or change it's meaning.
        Make sure all the details included in user's question is preserved in your version too.
        Reply just the rephrased question and no other information.
        """),
        ("human", state["question"]),
    ])

    print(f"* rephrase - {response}")

    # Querying vector store with rephrased question
    for doc in vector_store.similarity_search(response, k=6):
        all_docs[f"{doc.metadata["source"]}-{doc.metadata["page"]}"] = doc

    for doc in all_docs.values():
        print(doc)
        print("\n\n\n")

    return {
        "context": all_docs.values()
    }


def generate(state: State):
    # Combining docs fetched from vector store
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    # Creating LLM prompt
    prompt = ChatPromptTemplate([
        ("system", """
        You are a helpful assistant and your task is to help the user with cooking.
        To help with answering the user, this message includes the information from a few cooking books.
        If the information provided by the books is not relevant to the user's question or sufficiant, 
        then reply by saying that you don't have enough information.
        Only answer from the books and do not provide information that is not in the books.
        Answer to the user as if they are your friend. Don't use language that seem mechanical or robot like.

        -----------------------------------------
        Information from books: {context}    
        """),
        ("user", state["question"])
    ])

    # Creating LLM model
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.3, verbose=True)

    # Invoking the LLM model
    messages = prompt.invoke({"context": docs_content}).to_messages()
    response = llm.invoke(messages)
    return {
        "answer": response,
        "sources": [doc.metadata for doc in state["context"]]
    }


def build_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


if __name__ == "__main__":
    # Creating vector store
    vector_store = create_vector_store()

    # Constructing langgraph
    graph = build_graph()
    result = graph.invoke({
        "question": "what can I do with a pork tenderloin, cucumbers, potatoes, scallions, edamame, and rice?"
    })
    
    print()
    print(result["answer"])
    print("\nSouces:")
    for source in result["sources"]:
        print(f"* {source["source"]}, page {source["page"]}")