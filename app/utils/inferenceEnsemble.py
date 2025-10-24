import numpy as np
from .inferenceScripts.lgm_inference import process_url_with_heuristic_lightgbm
from .inferenceScripts.xgb_inference import process_url_with_heuristic_xgboost
from .inferenceScripts.rf_inference import process_url_with_heuristic_rf

def predict_ensemble(url: str):

    xgb_result = process_url_with_heuristic_xgboost(url)
    lgb_result = process_url_with_heuristic_lightgbm(url)
    rf_result  = process_url_with_heuristic_rf(url)

    # Extract per-model probability dicts
    xgb_probs = xgb_result["final_probabilities"]
    lgb_probs = lgb_result["final_probabilities"]
    rf_probs  = rf_result["final_probabilities"]

    # Ensure class order consistency
    classes = ['benign', 'defacement', 'malware', 'phishing']

    # Weighted soft voting
    weights = {"xgb": 0.3, "rf": 0.2, "lgb": 0.5}
    all_probs = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        all_probs[i] = (
            weights["xgb"] * xgb_probs[cls]
            + weights["rf"] * rf_probs[cls]
            + weights["lgb"] * lgb_probs[cls]
        )

    final_idx = np.argmax(all_probs)
    final_label = classes[final_idx]
    confidence = round(float(all_probs[final_idx]), 4)

    return {
        "url": url,
        "ensemble_prediction": final_label,
        "confidence": confidence,
        "probabilities": {cls: round(float(all_probs[i]), 4) for i, cls in enumerate(classes)},
        "model_details": {
            "xgboost": xgb_result,
            "lightgbm": lgb_result,
            "random_forest": rf_result,
        },
    }
