from datetime import datetime
from typing import Dict, Any, Optional

from antaryaami.ai_flow.models import RAGState, Metadata
from antaryaami.ai_flow.rag_graph import create_rag_graph


def run_rag_workflow(
    question: str,
    demographics: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> RAGState:
    """
    Run the RAG workflow for a given prediction question

    Args:
        question: The prediction question
        demographics: Optional demographic information
        source: Optional source of the question

    Returns:
        RAGState: The final state containing prediction and intermediate results
    """
    # Initialize state
    initial_state = RAGState(
        original_question=question,
        metadata=Metadata(
            created_at=datetime.now(), demographics=demographics or {}, source=source
        ),
    )

    # Create and run graph
    graph = create_rag_graph()
    final_state = graph.invoke(initial_state)

    return final_state
