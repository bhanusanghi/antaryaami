from typing import Annotated, Sequence, TypeVar
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import chain
from operator import itemgetter

from antaryaami.ai_flow.models import (
    RAGState,
    SearchQuery,
    SearchResult,
    PredictionResponse,
)

# Constants
QUERY_GENERATION_PROMPT = """Given a prediction question and its metadata, generate {num_queries} search queries that would help gather relevant information to make the prediction.

Question: {question}
Metadata: {metadata}

Generate specific and diverse search queries that will help gather information for making this prediction.
"""

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

# Initialize components
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
embeddings = OpenAIEmbeddings()
search_tool = DuckDuckGoSearchRun()
vector_store = Chroma(embedding_function=embeddings)


def generate_search_queries(state: RAGState, config: RunnableConfig) -> RAGState:
    """Generate search queries using LLM"""
    prompt = ChatPromptTemplate.from_template(QUERY_GENERATION_PROMPT)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {
            "question": state.original_question,
            "metadata": state.metadata.model_dump_json(),
            "num_queries": 3,  # Configurable
        }
    )

    # Parse queries from result
    queries = [line.strip() for line in result.split("\n") if line.strip()]
    state.search_queries = [SearchQuery(query=q) for q in queries]

    return state


def perform_search(state: RAGState, config: RunnableConfig) -> RAGState:
    """Execute search queries using search tools"""
    if not state.search_queries:
        state.error = "No search queries available"
        return state

    results = []
    for query in state.search_queries:
        try:
            search_result = search_tool.invoke(query.query)
            results.append(
                SearchResult(
                    content=search_result,
                    source="duckduckgo",
                    metadata={"query": query.query},
                )
            )
        except Exception as e:
            state.error = f"Search error: {str(e)}"

    state.search_results = results
    return state


def process_and_embed(state: RAGState, config: RunnableConfig) -> RAGState:
    """Process search results and store embeddings"""
    if not state.search_results:
        state.error = "No search results available"
        return state

    # Add documents to vector store
    texts = [result.content for result in state.search_results]
    metadatas = [result.metadata for result in state.search_results]

    vector_store.add_texts(texts=texts, metadatas=metadatas)
    return state


def retrieve_relevant(state: RAGState, config: RunnableConfig) -> RAGState:
    """Retrieve relevant chunks using vector similarity"""
    results = vector_store.similarity_search(
        state.original_question, k=4  # Configurable
    )

    state.relevant_chunks = [
        SearchResult(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            metadata=doc.metadata,
        )
        for doc in results
    ]

    return state


def make_prediction(state: RAGState, config: RunnableConfig) -> RAGState:
    """Make final prediction using LLM"""
    if not state.relevant_chunks:
        state.error = "No relevant chunks available"
        return state

    prompt = ChatPromptTemplate.from_template(PREDICTION_PROMPT)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke(
        {
            "question": state.original_question,
            "relevant_chunks": "\n".join(
                chunk.content for chunk in state.relevant_chunks
            ),
        }
    )

    # Parse prediction response
    lines = result.split("\n")
    state.prediction = PredictionResponse(
        yes_probability=float(lines[0].split(":")[1].strip()),
        no_probability=float(lines[1].split(":")[1].strip()),
        confidence_score=float(lines[2].split(":")[1].strip()),
        reasoning="\n".join(lines[4:]),
    )

    return state


def create_rag_graph() -> Graph:
    """Create the RAG workflow graph"""
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("search", perform_search)
    workflow.add_node("embed", process_and_embed)
    workflow.add_node("retrieve", retrieve_relevant)
    workflow.add_node("predict", make_prediction)

    # Add edges
    workflow.add_edge("generate_queries", "search")
    workflow.add_edge("search", "embed")
    workflow.add_edge("embed", "retrieve")
    workflow.add_edge("retrieve", "predict")

    # Set entry and end nodes
    workflow.set_entry_point("generate_queries")
    workflow.set_finish_point("predict")

    return workflow.compile()
