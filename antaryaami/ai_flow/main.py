from datetime import datetime
from antaryaami.ai_flow.run_rag import run_rag_workflow

# Run a prediction
result = run_rag_workflow(
    question="Will Bitcoin reach $150,000 by the end of april 2025?",
    demographics={"region": "global", "market": "crypto"},
    source="user_query",
    prediction_end_time=datetime(2025, 4, 30),
)

# Access prediction
if result.prediction:
    print(f"Yes Probability: {result.prediction.yes_probability}")
    print(f"No Probability: {result.prediction.no_probability}")
    print(f"Confidence: {result.prediction.confidence_score}")
    print(f"Reasoning: {result.prediction.reasoning}")
