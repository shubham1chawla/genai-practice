import logging
import os
from abc import ABC, abstractmethod

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_ollama import ChatOllama

from src.agent import build_agent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# Creating message classes for display
class BaseChatMessage(ABC):
    @property
    @abstractmethod
    def role(self) -> str:
        pass

    @abstractmethod
    def display(self):
        pass


class UserChatMessage(BaseChatMessage):
    def __init__(self, content: str):
        self.content = content
        st.session_state.messages.append(self)

    @property
    def role(self) -> str:
        return "user"

    def display(self):
        st.markdown(self.content)


class AIChatMessage(BaseChatMessage):
    def __init__(self):
        self.tool_messages = []
        self.ai_messages = []
        st.session_state.messages.append(self)

    @property
    def role(self):
        return "ai"

    def add_message(self, message: BaseMessage):
        if isinstance(message, ToolMessage):
            self.tool_messages.append(message)
        elif isinstance(message, AIMessage):
            self.ai_messages.append(message)

    def display(self):
        if not self.ai_messages:
            raise ValueError(f"You must `add_message` before displaying!")

        # Adding final AI message's content
        st.markdown(self.ai_messages[-1].content)

        # Checking if references are available
        if not self.tool_messages:
            return

        # Adding references
        with st.expander("Click to see tool messages"):
            tab_contents = [message.content for message in self.tool_messages]
            tabs = st.tabs([f"Tool {i + 1}: '{message.name}'" for i, message in enumerate(self.tool_messages)])
            for tab, tab_content in zip(tabs, tab_contents):
                with tab:
                    st.markdown(tab_content)


# Initialize session state to store conversation history
if "SESSION_STATE_SETUP_COMPLETE" not in st.session_state:
    st.session_state.messages = []
    st.session_state.input_disabled = False

    # Marking setup complete
    st.session_state["SESSION_STATE_SETUP_COMPLETE"] = True


def disable_input():
    st.session_state.input_disabled = True


# Set page configuration
st.set_page_config(
    page_title="Harry Potter AI Assistant",
    page_icon="✨",
    layout="wide"
)

# App title and description
st.title("Harry Potter Knowledge Assistant")
st.markdown("Ask questions about the Harry Potter universe!")

# Display conversation history
for chat_message in st.session_state.messages:
    chat_message: BaseChatMessage = chat_message
    with st.chat_message(chat_message.role):
        chat_message.display()

if prompt := st.chat_input(
        "Ask about Harry Potter...",
        disabled=st.session_state.input_disabled,
        on_submit=disable_input()
):
    # Display user message
    user_message = UserChatMessage(prompt)
    with st.chat_message(user_message.role):
        user_message.display()

    # Set up the agent
    logger.info("Setting up agent...")
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
    agent = build_agent(llm)
    inputs = {"messages": [("user", prompt)]}

    # Display ai message
    logger.info("Invoking agent...")
    ai_chat_message = AIChatMessage()
    with st.chat_message(ai_chat_message.role):
        message_placeholder = st.empty()

        # Streaming messages from the agent
        for value in agent.stream(inputs, stream_mode="values"):
            latest_message = value["messages"][-1]
            if isinstance(latest_message, ToolMessage):
                message_placeholder.markdown(f"_Looking at references..._")
            else:
                message_placeholder.markdown(f"_Thinking..._")
            ai_chat_message.add_message(latest_message)

        # Removing placeholder and displaying actual message
        message_placeholder.empty()
        ai_chat_message.display()

    logger.info("Completed agent call!")
    st.session_state.input_disabled = False
    st.rerun()
