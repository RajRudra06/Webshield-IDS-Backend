from pydantic import BaseModel
from typing import List,Dict, Optional

class URLFeatures(BaseModel):
    url: str

class URLRequest(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    threat_detected: bool
    confidence: float
    model_used: str
    category: str
    probabilities: Dict[str, float]
    reasons: Optional[List[str]] = []

class Feedback(BaseModel):
    url: str
    user_label: str  # "benign" or "malicious"
    model_pred: Optional[bool] = None
