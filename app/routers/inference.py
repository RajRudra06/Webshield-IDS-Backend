from fastapi import APIRouter, HTTPException
from ..models import URLFeatures, PredictionResponse
from ..utils.inferenceEnsemble import predict_ensemble

router = APIRouter()

@router.post("/", response_model=dict)  # Changed to dict for flexible response
def predict_url(data: URLFeatures):
   
    try:
        result = predict_ensemble(data.url)
        
        # Log summary
        final = result["final_decision"]
        print(f"[META-ENSEMBLE] URL: {data.url[:50]}... | "
              f"Final: {final['prediction']} ({final['confidence']:.2%}, {final['confidence_level']}) | "
              f"Threat: {final['threat_detected']}")
        
        return result
    
    except Exception as e:
        print(f"[ERROR] Prediction failed for {data.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")