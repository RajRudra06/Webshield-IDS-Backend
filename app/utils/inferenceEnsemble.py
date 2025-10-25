import joblib
import numpy as np

from .inferenceScripts.lgm_inference import process_url_with_heuristic_lightgbm
from .inferenceScripts.xgb_inference import process_url_with_heuristic_xgboost
from .inferenceScripts.rf_inference import process_url_with_heuristic_rf

# Updated path to include meta-learner
BASE_MODEL_PATH = "/Users/rudrarajpurohit/Desktop/Active Ps/webshield-backend/models/716k typosquatting/"
META_MODEL_PATH = BASE_MODEL_PATH + "meta_stacker.joblib"

# Load meta-learner at startup
meta_model = joblib.load(META_MODEL_PATH)

CLASSES = ['benign', 'defacement', 'malware', 'phishing']

def probs_to_vector(prob_dict):
    return np.array([prob_dict.get(c, 0.0) for c in CLASSES], dtype=float)


def engineer_meta_features(X_meta):
    
    n_samples = X_meta.shape[0]
    n_models = 3
    n_classes = 4
    
    # Reshape to (samples, models, classes)
    probas_3d = X_meta.reshape(n_samples, n_models, n_classes)
    
    features = []
    
    # 1. Mean probability per class (averaged across models)
    mean_prob_per_class = np.mean(probas_3d, axis=1)  # (n_samples, 4)
    features.append(mean_prob_per_class)
    
    # 2. Std deviation per class (model disagreement)
    std_prob_per_class = np.std(probas_3d, axis=1)  # (n_samples, 4)
    features.append(std_prob_per_class)
    
    # 3. Max probability across all predictions
    max_prob = np.max(probas_3d, axis=(1, 2)).reshape(-1, 1)
    features.append(max_prob)
    
    # 4. Min probability across all predictions
    min_prob = np.min(probas_3d, axis=(1, 2)).reshape(-1, 1)
    features.append(min_prob)
    
    # 5. Prediction agreement: which class do most models agree on?
    top_classes = np.argmax(probas_3d, axis=2)  # (n_samples, 3 models)
    agreement = np.array([
        np.max(np.bincount(row.astype(int))) / n_models 
        for row in top_classes
    ]).reshape(-1, 1)
    features.append(agreement)
    
    # 6. Entropy of mean predictions (uncertainty)
    mean_probs = np.mean(probas_3d, axis=1)
    epsilon = 1e-10
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1).reshape(-1, 1)
    features.append(entropy)
    
    # 7. Confidence margin (difference between top 2 probabilities)
    sorted_mean_probs = np.sort(mean_probs, axis=1)
    margin = (sorted_mean_probs[:, -1] - sorted_mean_probs[:, -2]).reshape(-1, 1)
    features.append(margin)
    
    # Concatenate all engineered features
    engineered = np.hstack(features)
    
    return np.hstack([X_meta, engineered])


def get_all_model_probs(url):
  
    xgb_res = process_url_with_heuristic_xgboost(url)
    lgb_res = process_url_with_heuristic_lightgbm(url)
    rf_res  = process_url_with_heuristic_rf(url)

    xgb_probs = probs_to_vector(xgb_res["final_probabilities"])
    lgb_probs = probs_to_vector(lgb_res["final_probabilities"])
    rf_probs  = probs_to_vector(rf_res["final_probabilities"])

    # Combine: [XGB_4, LGB_4, RF_4] = 12 features
    combined = np.concatenate([xgb_probs, lgb_probs, rf_probs])

    return {
        "combined_vector": combined,
        "xgb_result": {
            "prediction": xgb_res.get("final_prediction"),
            "probabilities": xgb_res.get("final_probabilities"),
            "confidence": max(xgb_res.get("final_probabilities", {}).values()),
        },
        "lgb_result": {
            "prediction": lgb_res.get("final_prediction"),
            "probabilities": lgb_res.get("final_probabilities"),
            "confidence": max(lgb_res.get("final_probabilities", {}).values()),
        },
        "rf_result": {
            "prediction": rf_res.get("final_prediction"),
            "probabilities": rf_res.get("final_probabilities"),
            "confidence": max(rf_res.get("final_probabilities", {}).values()),
        },
    }


def predict_ensemble(url: str):
   
    try:
        # Step 1: Get base model predictions
        results = get_all_model_probs(url)
        
        # Step 2: Prepare meta-features (12 base features)
        meta_features_raw = results["combined_vector"].reshape(1, -1)
        
        # Step 3: Engineer additional features (12 → 25 features)
        meta_features = engineer_meta_features(meta_features_raw)
        
        # Step 4: Meta-learner prediction
        proba = meta_model.predict_proba(meta_features)[0]
        idx = int(np.argmax(proba))
        label = CLASSES[idx]
        confidence = float(proba[idx])
        
        # Step 5: Determine confidence level
        if confidence > 0.85:
            confidence_level = "HIGH"
        elif confidence > 0.60:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return {
            "url": url,
            "meta_model_used": "meta_lightgbm_with_smote",  # Updated
            "ensemble_prediction": label,
            "ensemble_confidence": round(confidence, 4),
            "confidence_level": confidence_level,
            "ensemble_probabilities": {
                cls: round(float(proba[i]), 4) for i, cls in enumerate(CLASSES)
            },
            "base_models": {
                "xgboost": results["xgb_result"],
                "lightgbm": results["lgb_result"],
                "random_forest": results["rf_result"],
            },
            # Optional: Add model agreement info
            "meta_info": {
                "num_features_used": meta_features.shape[1],
                "base_features": 12,
                "engineered_features": 13,
            }
        }

    except Exception as e:
        raise RuntimeError(f"Meta-ensemble prediction failed: {e}")