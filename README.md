# Prediction Market AI Agent

A sophisticated AI agent built using LangChain and LangGraph frameworks to interact with and analyze prediction markets. This agent leverages large language models and blockchain integration to provide intelligent market analysis and interaction capabilities.

## Features

- Built with LangChain for robust AI capabilities
- Uses LangGraph for complex agent workflows and state management
- Blockchain integration for interacting with prediction markets
- Memory persistence and checkpointing
- Tool-based architecture for extensible functionality

## Prerequisites

- Python 3.10+
- Poetry for dependency management
- Access to OpenRouter API
- Web3 provider access (for blockchain interactions)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lang-starter
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install project dependencies:
```bash
poetry install
```

4. Create a `.env` file in the project root and add your configuration:
```bash
OPENAI_API_KEY=your_api_key_here
WEB3_PROVIDER_URL=your_web3_provider_url
PRIVATE_KEY=your_optional_private_key
```

## Setup Environment

1. Activate the Poetry virtual environment:
```bash
poetry shell
```

2. Verify the installation:
```bash
poetry env info
```

## Running the Project

[Add specific instructions for running your project here]

## Development

To add new dependencies:
```bash
poetry add package_name
```

To update dependencies:
```bash
poetry update
```

To run tests:
```bash
poetry run pytest
```




