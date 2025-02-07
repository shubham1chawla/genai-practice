#!.venv/bin/python

'''
Example inspired by - https://python.langchain.com/docs/integrations/vectorstores/chroma/
'''

from uuid import uuid4

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

MODEL_NAME = "llama3.2:3b"
COLLECTION_NAME = "example-1"

def add_documents(vector_store: Chroma):
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
        id=1,
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
        id=2,
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
        id=3,
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
        id=4,
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
        id=5,
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
        id=6,
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
        id=7,
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
        id=8,
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
        id=9,
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
        id=10,
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)


if __name__ == "__main__":
    # Creating vector store using Chroma db
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Adding documents to the vector store
    add_documents(vector_store)

    # Query directly - Similarity search
    '''
    results = vector_store.similarity_search(
        "LangChain provides abstractions to make working with LLMs easy",
        k=2,
        filter={"source": "tweet"},
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
    '''

    # Query directly - Similarity search with score
    '''
    results = vector_store.similarity_search_with_score(
        "Will it be hot tomorrow?", k=1, filter={"source": "news"}
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
    '''

    # Query directly - Search by vector
    '''
    results = vector_store.similarity_search_by_vector(
        embedding=embeddings.embed_query("I love green eggs and ham!"), k=1
    )
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")
    '''

    # Query by turning into retriever
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
    )
    results = retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
    print(results)