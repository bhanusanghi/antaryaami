# %%
from datetime import datetime
import os
import json
from typing import Annotated, Sequence, TypeVar
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables import chain
from langchain_community.tools.tavily_search import TavilySearchResults
import torch
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from IPython.display import Image, display
from langchain.tools import BaseTool
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage

from antaryaami.ai_flow.models import (
    RAGState,
    SearchQuery,
    SearchResult,
    PredictionResponse,
)

# %%

# Constants
QUERY_GENERATION_PROMPT = """Given a prediction question and its metadata, generate {num_queries} search queries that would help gather relevant information to make the prediction.

Question: {question}
Metadata: {metadata}

Generate specific and diverse search queries that will help gather information for making this prediction.
"""

TRANSFORM_QUESTION_PROMPT = """Transform the given prediction question into a more search-friendly format that will work better for semantic search and information retrieval.

Original Question: {question}

Guidelines:
1. Expand abbreviations and technical terms
2. Include relevant context and time frames
3. Break down complex questions into key aspects
4. Remove unnecessary words while keeping important context
5. Make it more neutral/objective in tone

Transform this into a search-optimized version that will help find relevant information.
Output only the transformed question without any explanation."""

PREDICTION_PROMPT = """Based on the original question and the relevant information gathered, provide a prediction with probabilities and confidence score.

Original Question: {question}
Relevant Information:
{relevant_chunks}

Provide:
1. Probability of Yes (between 0 and 1)
2. Probability of No (between 0 and 1)
3. Confidence Score (between 0 and 1)
4. Brief reasoning for the prediction

Note: Yes and No probabilities should sum to 1.
"""
# %%

# Read environment variables
open_router_api_key = os.getenv("OPENROUTER_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not open_router_api_key:
    raise ValueError("Please set OPENROUTER_API_KEY environment variable")
if not tavily_api_key:
    raise ValueError("Please set TAVILY_API_KEY environment variable")

# Initialize components
llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=open_router_api_key,
    temperature=0.2,
)

# Initialize search tools
duck_search_tool = DuckDuckGoSearchRun()
tavily_search_tool = TavilySearchResults(
    api_key=tavily_api_key, max_results=3, search_depth="advanced"
)

# Initialize HuggingFace embeddings with a smaller, efficient model
print("Initializing local HuggingFace embeddings...")

# Check for M1 GPU availability
if torch.backends.mps.is_available():
    device = "mps"  # M1 GPU
    print("Using M1 GPU (MPS) for embeddings")
    BATCH_SIZE = 64  # Good balance for 16GB M1
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
    print("Using NVIDIA GPU for embeddings")
    BATCH_SIZE = 32
else:
    device = "cpu"
    print("Using CPU for embeddings")
    BATCH_SIZE = 16

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, efficient model (80MB)
    model_kwargs={"device": device},
    encode_kwargs={"batch_size": BATCH_SIZE},  # Removed duplicate show_progress_bar
)

# Initialize search tools with proper Tool wrappers
search_tools = [
    Tool(
        name="duckduckgo_search",
        func=duck_search_tool.invoke,
        description="Search the web using DuckDuckGo. Use for general web searches and recent information.",
    ),
    # Tool(
    #     name="tavily_search",
    #     func=tavily_search_tool.invoke,
    #     description="Search using Tavily's advanced search API. Use for detailed research and analysis.",
    # ),
]
# - Use tavily_search for detailed research and analysis that requires deeper understanding
# Create agent with proper prompt and message formatting
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that uses search tools to find relevant information for making predictions.
    
            You have access to the following tool:
            - duckduckgo_search: Use for general web searches and recent information.

            DO NOT write explanations. execute a maximum of 1 search to reach conclusion.""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create tool calling agent instead of structured chat agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=search_tools,
    prompt=agent_prompt,
)

# Update agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=search_tools,
    verbose=True,
    handle_parsing_errors=False,
    max_iterations=1,
    max_execution_time=30,
)

# Initialize local vector store with persistence
persist_directory = "chroma_db"
os.makedirs(persist_directory, exist_ok=True)

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="prediction_rag",
)
print(f"Using Chroma in local mode with persistence at {persist_directory}")
# %%

# Initialize rich console for pretty printing
console = Console()


def log_node_execution(node_name: str, state: RAGState, message: str = None):
    """Helper function to log node execution details"""
    console.rule(f"[bold blue]Executing Node: {node_name}")

    if message:
        console.print(f"[yellow]{message}")

    # Log relevant state information based on node
    if node_name == "transform_question":
        console.print("[green]Input Question:", state.original_question)
        if state.transformed_question:
            console.print("[green]Transformed Question:", state.transformed_question)

    elif node_name == "generate_queries":
        if state.search_queries:
            console.print("[green]Generated Search Queries:")
            for i, query in enumerate(state.search_queries, 1):
                console.print(f"  {i}. {query.query}")

    elif node_name == "search":
        if state.search_results:
            console.print(f"[green]Search Results Found: {len(state.search_results)}")
            for i, result in enumerate(
                state.search_results[:2], 1
            ):  # Show first 2 results
                console.print(
                    Panel(
                        f"Source: {result.source}\nContent: {result.content[:200]}...",
                        title=f"Result {i}",
                    )
                )

    elif node_name == "embed":
        if state.search_results:
            console.print(f"[green]Embedding {len(state.search_results)} documents")

    elif node_name == "retrieve":
        if state.relevant_chunks:
            console.print(
                f"[green]Retrieved {len(state.relevant_chunks)} relevant chunks"
            )
            for i, chunk in enumerate(
                state.relevant_chunks[:2], 1
            ):  # Show first 2 chunks
                console.print(
                    Panel(
                        f"Source: {chunk.source}\nContent: {chunk.content[:200]}...",
                        title=f"Chunk {i}",
                    )
                )

    elif node_name == "predict":
        if state.prediction:
            console.print(
                Panel(
                    f"Yes Probability: {state.prediction.yes_probability:.2%}\n"
                    f"No Probability: {state.prediction.no_probability:.2%}\n"
                    f"Confidence: {state.prediction.confidence_score:.2%}\n"
                    f"Reasoning: {state.prediction.reasoning}",
                    title="Prediction Results",
                )
            )

    if state.error:
        console.print("[bold red]Error:", state.error)

    console.print()


# %%


# @todo -> this should also add relevant metadata. Like timeframe that might effect the question's search relevance. Add more dimensions to the question. Timeframe, region, market, etc. general scientific parameters if needed, analysis and tools for these analysis. Add all these dimenstions in the metadata.
# Keep evolving this metadata over time. compare with historical data.
# Find properties that would matter to this metadata.
# this should generate a type of Pydantic class temporary and be used in next steps. (Huggingface smol ILY. looking forward to you for this)


def transform_question(state: RAGState, config: RunnableConfig) -> RAGState:
    """Transform the original question into a search-optimized format"""
    log_node_execution(
        "transform_question", state, "Transforming question for better search..."
    )

    prompt = ChatPromptTemplate.from_template(TRANSFORM_QUESTION_PROMPT)

    chain = prompt | llm | StrOutputParser()

    try:
        transformed = chain.invoke({"question": state.original_question})
        state.transformed_question = transformed.strip()
        log_node_execution(
            "transform_question", state, "Question transformed successfully"
        )
    except Exception as e:
        state.error = f"Question transformation failed: {str(e)}"
        state.transformed_question = state.original_question  # Fallback to original
        log_node_execution("transform_question", state)

    return state


def generate_search_queries(state: RAGState, config: RunnableConfig) -> RAGState:
    """Generate search queries using LLM"""
    log_node_execution("generate_queries", state, "Generating search queries...")

    prompt = ChatPromptTemplate.from_template(QUERY_GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    question_for_queries = state.transformed_question or state.original_question

    result = chain.invoke(
        {
            "question": question_for_queries,
            "metadata": state.metadata.model_dump_json(),
            "num_queries": 3,
        }
    )

    queries = [line.strip() for line in result.split("\n") if line.strip()]
    state.search_queries = [SearchQuery(query=q) for q in queries]

    log_node_execution("generate_queries", state)
    return state


def perform_search(state: RAGState, config: RunnableConfig) -> RAGState:
    """Execute search queries using agent executor"""
    log_node_execution("search", state, "Executing searches...")

    if not state.search_queries:
        state.error = "No search queries available"
        log_node_execution("search", state)
        return state

    results = []
    for query in state.search_queries:
        try:
            task = f"""Search for information to help predict: {query.query}
            Focus on finding factual, relevant information that will help make an accurate prediction.
            """

            # Execute agent with tool calling
            agent_response = agent_executor.invoke(
                {
                    "input": task,
                    "chat_history": [
                        HumanMessage(content=state.original_question),
                        AIMessage(content=f"Searching for: {query.query}"),
                    ],
                },
                config={"tags": ["search"]},
            )

            # Handle both normal output and timeout/iteration limit cases
            if isinstance(agent_response, dict):
                if "output" in agent_response:
                    content = str(agent_response["output"])
                elif "intermediate_steps" in agent_response:
                    # Extract results from intermediate steps if available
                    content = "\n".join(
                        step[1]
                        for step in agent_response["intermediate_steps"]
                        if isinstance(step, tuple) and len(step) > 1
                    )
                else:
                    content = "No results found"

                if content and content != "No results found":
                    search_result = SearchResult(
                        content=content,
                        source="duckduckgo_search",
                        metadata={
                            "query": query.query,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    results.append(search_result)
            else:
                console.print(
                    f"[yellow]Warning: Unexpected response type for query: {query.query}"
                )

        except Exception as e:
            console.print(f"[bold red]Error in search: {str(e)}")
            # Try to extract any partial results from the error state
            if hasattr(e, "intermediate_steps"):
                try:
                    content = "\n".join(
                        step[1]
                        for step in getattr(e, "intermediate_steps", [])
                        if isinstance(step, tuple) and len(step) > 1
                    )
                    if content:
                        search_result = SearchResult(
                            content=content,
                            source="duckduckgo_search",
                            metadata={
                                "query": query.query,
                                "partial": True,
                                "error": str(e),
                            },
                        )
                        results.append(search_result)
                except Exception:
                    pass
            continue

    if results:
        state.search_results = results
    else:
        state.error = "No valid search results obtained"

    log_node_execution("search", state)
    return state


def process_and_embed(state: RAGState, config: RunnableConfig) -> RAGState:
    """Process search results and store embeddings"""
    log_node_execution("embed", state, "Processing and embedding search results...")

    if not state.search_results:
        state.error = "No search results available"
        log_node_execution("embed", state)
        return state

    # Add documents to vector store
    texts = [result.content for result in state.search_results]
    metadatas = [result.metadata for result in state.search_results]

    vector_store.add_texts(texts=texts, metadatas=metadatas)

    log_node_execution("embed", state)
    return state


def retrieve_relevant(state: RAGState, config: RunnableConfig) -> RAGState:
    """Retrieve relevant chunks using vector similarity"""
    log_node_execution("retrieve", state, "Retrieving relevant information...")

    search_query = state.transformed_question or state.original_question
    results = vector_store.similarity_search(search_query, k=4)

    state.relevant_chunks = [
        SearchResult(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            metadata=doc.metadata,
        )
        for doc in results
    ]

    log_node_execution("retrieve", state)
    return state


def make_prediction(state: RAGState, config: RunnableConfig) -> RAGState:
    """Make final prediction using LLM"""
    log_node_execution("predict", state, "Generating prediction...")

    if not state.relevant_chunks:
        state.error = "No relevant chunks available"
        log_node_execution("predict", state)
        return state

    prompt = ChatPromptTemplate.from_template(PREDICTION_PROMPT)
    formatted_chunks = "\n\n".join(
        f"Source: {chunk.source}\nContent: {chunk.content}"
        for chunk in state.relevant_chunks
    )

    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke(
            {"question": state.original_question, "relevant_chunks": formatted_chunks}
        )

        lines = [line.strip() for line in result.split("\n") if line.strip()]

        try:
            yes_prob = float(lines[0].split(":")[-1].strip().replace("%", "")) / 100
            no_prob = float(lines[1].split(":")[-1].strip().replace("%", "")) / 100
            confidence = float(lines[2].split(":")[-1].strip().replace("%", "")) / 100

            total = yes_prob + no_prob
            if total != 0:
                yes_prob = yes_prob / total
                no_prob = no_prob / total

            state.prediction = PredictionResponse(
                yes_probability=max(0.0, min(1.0, yes_prob)),
                no_probability=max(0.0, min(1.0, no_prob)),
                confidence_score=max(0.0, min(1.0, confidence)),
                reasoning=(
                    "\n".join(lines[4:])
                    if len(lines) > 4
                    else "No detailed reasoning provided"
                ),
            )

        except (IndexError, ValueError) as e:
            state.error = (
                f"Failed to parse prediction values: {str(e)}\nRaw response: {result}"
            )

    except Exception as e:
        state.error = f"LLM prediction failed: {str(e)}"

    log_node_execution("predict", state)
    return state


# %%


def create_rag_graph():
    """Create the RAG workflow graph"""
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("transform_question", transform_question)
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("search", perform_search)
    workflow.add_node("embed", process_and_embed)
    workflow.add_node("retrieve", retrieve_relevant)
    workflow.add_node("predict", make_prediction)

    # Add edges including START and END
    workflow.add_edge(START, "transform_question")
    workflow.add_edge("transform_question", "generate_queries")
    workflow.add_edge("generate_queries", "search")
    workflow.add_edge("search", "embed")
    workflow.add_edge("embed", "retrieve")
    workflow.add_edge("retrieve", "predict")
    workflow.add_edge("predict", END)

    resp = workflow.compile()

    try:
        display(Image(resp.get_graph().draw_mermaid_png()))
    except Exception:
        pass

    return resp


# %%
