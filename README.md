# Antaryaami RAG System

A Retrieval-Augmented Generation (RAG) system for making predictions based on real-time information retrieval.

## Features

- Query generation using LLMs
- Multi-tool search capabilities (DuckDuckGo)
- Vector store for efficient information retrieval (Chroma)
- LangGraph-based workflow orchestration
- Structured prediction output with probabilities and confidence scores

## Installation

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY=your_openai_api_key
```

## Usage

```python
from antaryaami.ai_flow.run_rag import run_rag_workflow

# Run a prediction
result = run_rag_workflow(
    question="Will Bitcoin reach $100,000 by the end of 2024?",
    demographics={"region": "global", "market": "crypto"},
    source="user_query"
)

# Access prediction
if result.prediction:
    print(f"Yes Probability: {result.prediction.yes_probability}")
    print(f"No Probability: {result.prediction.no_probability}")
    print(f"Confidence: {result.prediction.confidence_score}")
    print(f"Reasoning: {result.prediction.reasoning}")
```

## Architecture

The system follows a graph-based workflow:

1. Query Generation Node: Generates search queries based on the prediction question
2. Search Node: Executes searches using configured tools
3. Embedding Node: Processes and embeds search results
4. Retrieval Node: Finds relevant information using vector similarity
5. Prediction Node: Makes final prediction using LLM

## Development

- Format code: `poetry run black .`
- Sort imports: `poetry run isort .`
- Type checking: `poetry run mypy .`
- Run tests: `poetry run pytest`

## License

MIT License




