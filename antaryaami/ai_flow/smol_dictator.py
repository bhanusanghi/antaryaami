from antaryaami.ai_flow.smol_predictoor import process_prediction_request
from antaryaami.ai_flow.models import Metadata
from datetime import datetime

result = process_prediction_request(
    "Fed decision in January 2025? will they increase interest rates by +25bps?"
)

print(result)
