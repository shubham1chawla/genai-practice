#!.venv/bin/python

from langchain.output_parsers.fix import OutputFixingParser
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing_extensions import List, TypedDict, Optional

from src import constants
from src.books import Book
from src.db import create_vector_store
from src.logger import logger, loggraph


class State(TypedDict):
    question: str
    rephrased_question: str
    relevant_books: List[Book]
    context: List[Document]
    answer: str
    sources: List[Document]
    refine_notes: Optional[str]
    refine_count: int


class IdentifyResponse(BaseModel):
    relevant: bool = Field(description="The book contains relevant information to answer user's question.")


@loggraph
def identify(state: State):
    # Creating prompt & chain to short-list the book
    prompt = ChatPromptTemplate([
        ("system", """
        You are an helpful assistant who is expert at Harry Potter series.
        Your task is to tell whether the book {book_order} - "{book_name}" 
        may contain relevant information that may answer user's question.
        
        {format_instructions}
        """),
        ("human", "{question}"),
    ])
    parser = OutputFixingParser.from_llm(
        llm=json_llm,
        parser=PydanticOutputParser(pydantic_object=IdentifyResponse),
        max_retries=5,
    )
    chain = prompt | json_llm | parser

    # Identifying relevant books
    relevant_books = []
    for i, book in enumerate(Book.load(constants.BOOKS_JSON_PATH)):
        response: IdentifyResponse = chain.invoke({
            "question": state["question"],
            "book_order": i + 1,
            "book_name": book.name,
            "format_instructions": parser.get_format_instructions(),
        })
        if response.relevant:
            logger.info(f"Adding '{book.name}' to relevant books.")
            relevant_books.append(book)

    return {
        "relevant_books": relevant_books,
    }


@loggraph
def rephrase(state: State):
    # Setting up prompt and chain
    prompt = ChatPromptTemplate([
        ("system", """
        You are an expert at understanding the user's question on the Harry Potter series
        , and are tasked to elaborate user's simple question into a more specific & directed question.
        Your rephrased question shouldn't distort user's question or change it's meaning.
        Make sure all the details included in user's question is preserved in your version too.
        Optionally, this message would have additional notes for your to consider while you are
        rephrasing the user's question.
        Reply just the rephrased question and no other information.
         
        Optional additional notes: {refine_notes}
        """),
        ("human", "{question}"),
    ])
    parser = StrOutputParser()
    chain = prompt | llm | parser

    # Invoking LLM to extract response
    response = chain.invoke({
        "question": state["question"],
        "refine_notes": state.get("refine_notes", ""),
    })
    logger.info(f"Rephrased Question: {response}")

    return {
        "rephrased_question": response,
    }


@loggraph
def retrieve(state: State):
    search_kwargs = {
        "k": constants.K_VALUE,
    }

    # Adding relevant books to filter
    if relevant_books := state.get("relevant_books", []):
        search_kwargs.update({
            "filter": {
                "book.id": {
                    "$in": [book.id for book in relevant_books]
                }
            }
        })

    # Creating retriever for fetching documents
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    # Fetching documents from vector store
    documents = {doc.metadata["id"]: doc for doc in state.get("sources", [])}
    for doc in retriever.invoke(state["rephrased_question"]):
        doc_info = f"{doc.metadata["book.name"]}:{doc.metadata["page"]}"
        logger.info(doc_info)
        documents[doc.metadata["id"]] = doc
    logger.info(f"Total documents loaded: {len(documents)}")

    return {
        "context": "\n\n".join(doc.page_content for doc in documents.values()),
        "sources": documents.values(),
    }


class PurgeResponse(BaseModel):
    include: bool = Field(description="the text from the book is relevant")


@loggraph
def purge(state: State):
    # Creating a prompt to remove unnessary documents
    prompt = ChatPromptTemplate([
        ("system", """
        You are a helpful assistant and an expert at Harry Potter books.
        This message contains text from the Harry Potter books, which may
        be relevant to answer user's question.
        Your task is to tell whether the provided text is relevant to
        answer the user's question or not.
         
        {document}
        {format_instructions}
        """),
        ("human", "{question}"),
    ])
    parser = OutputFixingParser.from_llm(
        llm=json_llm,
        parser=PydanticOutputParser(pydantic_object=PurgeResponse),
        max_retries=5,
    )
    chain = prompt | json_llm | parser

    # Filtering documents
    relevant_documents = []
    for doc in state["sources"]:
        response: PurgeResponse = chain.invoke({
            "question": state["question"],
            "document": doc.page_content,
            "format_instructions": parser.get_format_instructions(),
        })
        if not response.include:
            logger.info(f"Removing document {doc.metadata["book.name"]}:{doc.metadata["page"]}")
        else:
            relevant_documents.append(doc)

    return {
        "context": "\n\n".join(doc.page_content for doc in relevant_documents),
        "sources": relevant_documents,
    }


class ReviewResponse(BaseModel):
    approved: bool = Field(description="true if the references provided is enough to answer user's question")


@loggraph
def review(state: State):
    # Bypassing refining process
    if state.get("refine_count", 0) <= 0:
        logger.info("Exhausted all refining retries, proceeding to generate answer...")
        return "generate"

    # Creating chain to figure out whether llm has enough information to answer
    prompt = ChatPromptTemplate([
        ("system", """
        This message contains references from the Harry Potter books.
        Your task is to tell whether these references provide enough context to
        answer the user's question and return your assessment as the instructions mentioned below.
        
        {format_instructions}
        
        --------------------------------------
        Context from the books: {context}
        """),
        ("human", "{question}")
    ])
    parser = OutputFixingParser.from_llm(
        llm=json_llm,
        parser=PydanticOutputParser(pydantic_object=ReviewResponse),
        max_retries=5,
    )
    chain = prompt | json_llm | parser
    response: ReviewResponse = chain.invoke({
        "question": state["question"],
        "context": state["context"],
        "format_instructions": parser.get_format_instructions(),
    })

    # Checking if llm can proceed to answer to the user
    if response.approved:
        logger.info("No need to refine prompts, proceeding to generate answer...")
        return "generate"

    logger.info("Required additional context, refining prompts to fetch more documents...")
    return "refine"


@loggraph
def refine(state: State):
    # Creating chain to refining prompt to fetch relevant sources
    prompt = ChatPromptTemplate([
        ("system", """
        A user has asked a question regarding the Harry Potter series.
        This message will contain a few references from the books which can help answer the question.
        However, there is a need to fetch more references from the books to generate a well-rounded answer.
        Your task is to evaluate the information fetched from the books so far, and provide additional
        informtion that the user's question should have so that the updated question can find better references
        from the books.
        Reply with only the additional information you suggest to be included in the question.
         
        Information from the books fetched so far: {context}
        """),
        ("human", "{question}")
    ])
    parser = StrOutputParser()
    chain = prompt | llm | parser

    # Fetching response form the llm to expand user's question
    response = chain.invoke({
        "context": state["context"],
        "question": state["question"],
    })
    logger.info(f"Refining Notes: {response}")

    return {
        "refine_notes": response,
        "refine_count": state.get("refine_count", 0) - 1,
    }


@loggraph
def generate(state: State):
    # Creating chain and generating answer
    prompt = ChatPromptTemplate([
        ("system", """
        You are an expert at Harry Potter series and your task is to answer user's question.
        To help with answering the user, this message includes the information from the Harry Potter books.
        If the information provided by the books is not relevant to the user's question or sufficiant, 
        then reply by saying that you don't have enough information.
        Only answer from the information provided in this message.
        You are encouraged to quote from references that are provided to you, if that helps create an engaging answer.
        Answer to the user as if they are your friend. Don't use language that seem mechanical or robot like.

        -----------------------------------------
        Information from books: {context}    
        """),
        ("user", state["question"])
    ])
    parser = StrOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({"context": state["context"]})

    return {
        "answer": response,
    }


def build_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(State)

    # Adding nodes to the graph
    graph_builder.add_node("identify", identify)
    graph_builder.add_node("rephrase", rephrase)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("purge", purge)
    graph_builder.add_node("refine", refine)
    graph_builder.add_node("generate", generate)

    # Adding conditional node
    graph_builder.add_conditional_edges(
        "purge",
        review,
        {
            "refine": "refine",
            "generate": "generate",
        }
    )

    # Adding edges
    graph_builder.add_edge(START, "identify")
    graph_builder.add_edge("identify", "rephrase")
    graph_builder.add_edge("rephrase", "retrieve")
    graph_builder.add_edge("retrieve", "purge")
    graph_builder.add_edge("refine", "identify")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()


if __name__ == "__main__":
    # Creating vector store & retriever
    vector_store = create_vector_store(
        constants.EMBEDDINGS_MODEL_NAME,
        constants.COLLECTION_NAME,
        constants.VECTOR_STORE_DIR,
    )

    # Creating llms
    llm = ChatOllama(model=constants.LLM_MODEL_NAME, temperature=0.3)
    json_llm = ChatOllama(model=constants.LLM_MODEL_NAME, temperature=0, format="json")

    # Constructing langgraph
    graph = build_graph()
    result = graph.invoke({
        # "question": "who put harry's name in the goblet of fire?",
        # "question": "What was harry's prophecy in the book order of phoenix?",
        # "question": "What were the names of professors who taught the defense against the darks arts at hogwarts?",
        # "question": "When harry was stuck in the dark forest hiding from dementors in the third book, who casted the petronus spell?",
        # "question": "Who helped harry get the secret from the orb after the dragon fight in the goblet of fire?",
        # "question": "why was dobby hiding in harry's room in the book chamber of secrets?",
        # "question": "who murdered professor dumbledore?",
        # "question": "why did snape and draco's mother enter into an unbreakable vow?",
        "question": "what did ollivander tell harry about his wand in the 1st book?",
        "refine_count": constants.REFINE_COUNT,
    })

    print()
    print(result["answer"])
    print("\nSouces:")
    for doc in result["sources"]:
        metadata = doc.metadata
        print(f"* {metadata["book.name"]}, page {metadata["page"]}")
