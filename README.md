# Generative AI Practice Repository

A collection of examples demonstrating various applications of Generative AI using LangChain, Ollama, and other related technologies.

## Project Overview

This repository contains several examples that showcase different aspects of working with Generative AI:

- **Example 0**: Basic integration of Ollama with LangChain
- **Example 1**: Document embedding and retrieval using ChromaDB
- **Example 2**: Recipe generation RAG (Retrieval-Augmented Generation) application using cooking books
- **Example 3**: Advanced RAG application using Harry Potter books, featuring stateful applications with Langgraph
- **Example 4**: Knowledge Graph generator using Harry Potter books, featuring graph database Neo4j (Check example-4's [README](./example-4/README.md))

## Prerequisites

- Python 3.12 or higher
- Poetry (Python package manager)
- Ollama

## Installation

1. First, install Ollama by following the instructions for your operating system. Visit [Ollama's official website](https://ollama.ai/download) for Windows installation instructions.

2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/genai-practice.git
   cd genai-practice
   ```

4. Install project dependencies using Poetry:
   ```bash
   poetry install
   ```

## Project Structure

```
.
├── example-0/  # Ollama + LangChain boilerplate
├── example-1/  # ChromaDB document embedding demo
├── example-2/  # Recipe Generation RAG
├── example-3/  # Harry Potter RAG with Langgraph
└── example-4/  # Knowledge Graph Builder with Neo4j
```

## Usage

1. Start the Ollama server:
   ```bash
   ollama serve
   ```

2. Pull the required model (e.g., llama2):
   ```bash
   ollama pull llama2
   ```

3. Activate the Poetry virtual environment:
   ```bash
   poetry shell
   ```

4. Navigate to any example directory and run the Python scripts:
   ```bash
   cd example-0
   python main.py
   ```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
