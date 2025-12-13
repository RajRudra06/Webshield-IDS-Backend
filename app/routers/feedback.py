from fastapi import APIRouter, Body
from ..models import Feedback

router = APIRouter()
_FEEDBACKS = []

@router.post("/", response_model=dict)
def submit_feedback(data: Feedback = Body(...)):
    _FEEDBACKS.append(data.dict())
    return {"status": "received", "count": len(_FEEDBACKS)}
