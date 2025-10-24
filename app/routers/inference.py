from fastapi import APIRouter, HTTPException
from ..models import URLFeatures, PredictionResponse
from ..utils.inferenceEnsemble import predict_ensemble

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
def predict_url(data: URLFeatures):
    """
    Run ensemble prediction on a given URL and return threat classification results.
    """
    try:
        result = predict_ensemble(data.url)

        final_label = result["ensemble_prediction"]      # 'benign', 'phishing', etc.
        confidence = result["confidence"]                # numeric probability
        model_used = "ensemble_RF_XGB_LGBM"
        is_threat = final_label != "benign"

        print({
            "threat_detected": is_threat,
            "confidence": confidence,
            "model_used": model_used,
            "category": final_label,                    # <-- added this
            "probabilities": result["probabilities"],   # optional detailed breakdown
        })

        # Include full label (category) in the response
        return {
            "threat_detected": is_threat,
            "confidence": confidence,
            "model_used": model_used,
            "category": final_label,                    # <-- added this
            "probabilities": result["probabilities"],   # optional detailed breakdown
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
