from fastapi import APIRouter, HTTPException
from ..models import URLFeatures, PredictionResponse
from ..utils.inferenceEnsemble import predict_ensemble

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
def predict_url(data: URLFeatures):
    """
    Run meta-ensemble prediction on a given URL and return threat classification.
    
    Now uses improved meta-learner with:
    - SMOTE balancing
    - 13 engineered features
    - LightGBM meta-classifier
    - 98.45% accuracy
    """
    try:
        result = predict_ensemble(data.url)

        final_label = result["ensemble_prediction"]
        confidence = result["ensemble_confidence"]
        confidence_level = result["confidence_level"]
        model_used = result["meta_model_used"]
        is_threat = final_label != "benign"

        # Log for monitoring
        print(f"[META-ENSEMBLE] URL: {data.url[:50]}... | "
              f"Prediction: {final_label} | "
              f"Confidence: {confidence:.2%} ({confidence_level}) | "
              f"Threat: {is_threat}")

        return {
            "threat_detected": is_threat,
            "confidence": confidence,
            "confidence_level": confidence_level,  # NEW: HIGH/MEDIUM/LOW
            "model_used": model_used,
            "category": final_label,
            "probabilities": result["ensemble_probabilities"],
            "base_models": result.get("base_models"),  # Optional: include base results
        }
    
    except Exception as e:
        # Log error for debugging
        print(f"[ERROR] Prediction failed for {data.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")