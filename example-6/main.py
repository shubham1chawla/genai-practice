import logging
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from src.agent import build_agent


def invoke_agent(prompt: str):
    # Setting up agent
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
    agent = build_agent(llm)
    inputs = {"messages": [("user", prompt)]}

    # Printing message stream
    for value in agent.stream(inputs, stream_mode="values"):
        latest_message = value["messages"][-1]
        if isinstance(latest_message, tuple):
            print(latest_message)
        else:
            latest_message.pretty_print()


def main():
    # prompt = "Is there a book that contains goblet in it?"
    # prompt = "What kind of character was dobby in chamber secrets book?"
    # prompt = "What are horcrux?"
    # prompt = "How did mcgonagall treat her students?"
    # prompt = "How would describe chemistry between Harry and Dobby?"
    # prompt = "What was the rivalry between ron and malfoy?"
    # prompt = "How would you describe Draco Malfoy character in deathly hallows?"
    prompt = "What kind of creatures are described in the books?"

    # Invoking agent
    invoke_agent(prompt)


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Muting packages
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main()
