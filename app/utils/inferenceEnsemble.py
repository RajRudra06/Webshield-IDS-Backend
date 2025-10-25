import joblib
import numpy as np
from pathlib import Path

import numpy as np
from .inferenceScripts.lgm_inference import process_url_with_heuristic_lightgbm
from .inferenceScripts.xgb_inference import process_url_with_heuristic_xgboost
from .inferenceScripts.rf_inference import process_url_with_heuristic_rf

META_MODEL_PATH = "/Users/rudrarajpurohit/Desktop/Active Ps/webshield-backend/models/716k typosquatting/"
meta_model = joblib.load(META_MODEL_PATH)

CLASSES = ['benign', 'defacement', 'malware', 'phishing']

def probs_to_vector(prob_dict):
    return np.array([prob_dict.get(c, 0.0) for c in CLASSES], dtype=float)

def get_all_model_probs(url):

    xgb_res = process_url_with_heuristic_xgboost(url)
    lgb_res = process_url_with_heuristic_lightgbm(url)
    rf_res  = process_url_with_heuristic_rf(url)

    xgb_probs = probs_to_vector(xgb_res["final_probabilities"])
    lgb_probs = probs_to_vector(lgb_res["final_probabilities"])
    rf_probs  = probs_to_vector(rf_res["final_probabilities"])

    combined = np.concatenate([xgb_probs, lgb_probs, rf_probs])

    return {
        "combined_vector": combined,
        "xgb_result": {
            "prediction": xgb_res.get("final_prediction"),
            "probabilities": xgb_res.get("final_probabilities"),
        },
        "lgb_result": {
            "prediction": lgb_res.get("final_prediction"),
            "probabilities": lgb_res.get("final_probabilities"),
        },
        "rf_result": {
            "prediction": rf_res.get("final_prediction"),
            "probabilities": rf_res.get("final_probabilities"),
        },
    }


def predict_ensemble(url: str):
   
    try:
        results = get_all_model_probs(url)
        meta_features = results["combined_vector"].reshape(1, -1)

        proba = meta_model.predict_proba(meta_features)[0]
        idx = int(np.argmax(proba))
        label = CLASSES[idx]
        confidence = float(proba[idx])

        return {
            "url": url,
            "meta_model_used": "meta_logistic_regression",
            "ensemble_prediction": label,
            "ensemble_confidence": round(confidence, 4),
            "ensemble_probabilities": {
                cls: round(float(proba[i]), 4) for i, cls in enumerate(CLASSES)
            },
            "base_models": {
                "xgboost": results["xgb_result"],
                "lightgbm": results["lgb_result"],
                "random_forest": results["rf_result"],
            },
        }

    except Exception as e:
        raise RuntimeError(f"Meta-ensemble failed: {e}")
