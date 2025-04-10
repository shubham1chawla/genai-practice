from datetime import datetime
from typing import Optional

from langchain import agents, hub
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph import prebuilt


@tool
def get_current_date_time(fmt: str) -> str:
    """
    Returns current system date and time in the format provided.

    :param fmt: format (example: %Y-%m-%d %H:%M:%S for entire date and time)
    :return: formatted current time
    """
    time = datetime.now()
    return time.strftime(fmt)


@tool
def get_weather(city: str) -> str:
    """
    Returns the weather information of nyc or sf only.

    :param city: nyc or sf
    :return: City's weather
    """
    if city == "nyc":
        return "cloudy"
    elif city == "sf":
        return "sunny"
    else:
        return f"Invalid input! Only give nyc or sf as action input!"


def run_legacy(query: str, verbose=False) -> str:
    llm = ChatOllama(model="phi4:14b", temperature=0)
    tools = [get_current_date_time, get_weather]
    executor = agents.AgentExecutor(
        agent=agents.create_react_agent(llm, tools, hub.pull("hwchase17/react")),
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
    )
    result = executor.invoke({"input": query})
    return result.get("output")


def run_modern(query: str, verbose=False) -> str:
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    tools = [get_current_date_time, get_weather]
    graph = prebuilt.create_react_agent(llm, tools=tools)

    last_message: Optional[AIMessage] = None

    def print_stream(stream):
        for s in stream:
            message = s["messages"][-1]
            nonlocal last_message
            last_message = message
            if not verbose:
                continue

            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    inputs = {"messages": [("user", query)]}
    print_stream(graph.stream(inputs, stream_mode="values"))
    return last_message.content


if __name__ == "__main__":
    user_query = "what's the current time in hours and minutes? Also, what's weather in nyc?"

    print("Running legacy: ")
    print(run_legacy(user_query, False))

    print("Running modern: ")
    print(run_modern(user_query, False))
