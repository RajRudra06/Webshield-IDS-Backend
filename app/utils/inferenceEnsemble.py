import joblib
import numpy as np
from pathlib import Path

from .inferenceScripts.lgm_inference import process_url_with_heuristic_lightgbm
from .inferenceScripts.xgb_inference import process_url_with_heuristic_xgboost
from .inferenceScripts.rf_inference import process_url_with_heuristic_rf

from .rl_Implementation.rl_agent import rl_agent
from .rl_Implementation.rl_state_extractor import extract_rl_state, discretize_state
from .rl_Implementation.rl_action_executor import execute_rl_action
from .rl_Implementation.prediction_buffer import prediction_buffer

BASE_DIR = Path(__file__).resolve().parents[2]
META_MODEL_PATH = BASE_DIR / "models" / "716k typosquatting" / "meta_stacker.joblib"

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

    def safe_get(res, key_main, key_fallback=None, default=None):
        if key_main in res:
            return res[key_main]
        elif key_fallback and key_fallback in res:
            return res[key_fallback]
        return default

    xgb_pred = safe_get(xgb_res, "final_prediction", "model_prediction")
    lgb_pred = safe_get(lgb_res, "final_prediction", "model_prediction")
    rf_pred  = safe_get(rf_res, "final_prediction", "model_prediction")

    xgb_probs = safe_get(xgb_res, "final_probabilities", "model_probabilities", {})
    lgb_probs = safe_get(lgb_res, "final_probabilities", "model_probabilities", {})
    rf_probs  = safe_get(rf_res, "final_probabilities", "model_probabilities", {})

    xgb_vec = probs_to_vector(xgb_probs)
    lgb_vec = probs_to_vector(lgb_probs)
    rf_vec  = probs_to_vector(rf_probs)

    return {
        "combined_vector": np.concatenate([xgb_vec, lgb_vec, rf_vec]),
        "xgb_result": {"final_prediction": xgb_pred, "final_probabilities": xgb_probs},
        "lgb_result": {"final_prediction": lgb_pred, "final_probabilities": lgb_probs},
        "rf_result":  {"final_prediction": rf_pred, "final_probabilities": rf_probs}
    }



def predict_ensemble(url: str):
   
    try:
        results = get_all_model_probs(url)

                # ✅ Consensus override: if all base models agree, skip meta-model
        base_preds = [
            results["lgb_result"]["final_prediction"],
            results["xgb_result"]["final_prediction"],
            results["rf_result"]["final_prediction"]
        ]

        if len(set(base_preds)) == 1:  # all 3 identical
            unanimous_label = base_preds[0]
            unanimous_conf = max(
                max(results["lgb_result"]["final_probabilities"].values()),
                max(results["xgb_result"]["final_probabilities"].values()),
                max(results["rf_result"]["final_probabilities"].values())
            )

            return {
                "url": url,
                "final_decision": {
                    "prediction": unanimous_label,
                    "confidence": round(unanimous_conf, 4),
                    "confidence_level": (
                        "HIGH" if unanimous_conf > 0.85
                        else "MEDIUM" if unanimous_conf > 0.60
                        else "LOW"
                    ),
                    "threat_detected": unanimous_label != "benign",
                    "model_used": "base_consensus"
                },
                "note": "All base models agreed; meta-model skipped.",
                "model_predictions": {
                    "lightgbm": {
                        "prediction": results["lgb_result"]["final_prediction"],
                        "confidence": round(max(results["lgb_result"]["final_probabilities"].values()), 4),
                        "probabilities": results["lgb_result"]["final_probabilities"],
                        "threat_detected": results["lgb_result"]["final_prediction"] != "benign"
                    },
                    "xgboost": {
                        "prediction": results["xgb_result"]["final_prediction"],
                        "confidence": round(max(results["xgb_result"]["final_probabilities"].values()), 4),
                        "probabilities": results["xgb_result"]["final_probabilities"],
                        "threat_detected": results["xgb_result"]["final_prediction"] != "benign"
                    },
                    "random_forest": {
                        "prediction": results["rf_result"]["final_prediction"],
                        "confidence": round(max(results["rf_result"]["final_probabilities"].values()), 4),
                        "probabilities": results["rf_result"]["final_probabilities"],
                        "threat_detected": results["rf_result"]["final_prediction"] != "benign"
                    }
                },
                "meta_info": {
                    "num_features_used": 0,
                    "base_features": 12,
                    "engineered_features": 0,
                    "models_used": ["LightGBM", "XGBoost", "RandomForest"]
                }
            }

        
        meta_features_raw = results["combined_vector"].reshape(1, -1)
        
        meta_features = engineer_meta_features(meta_features_raw)
        
        proba = meta_model.predict_proba(meta_features)[0]
        idx = int(np.argmax(proba))
        meta_label = CLASSES[idx]
        meta_confidence = float(proba[idx])
        
        if meta_confidence > 0.85:
            confidence_level = "HIGH"
        elif meta_confidence > 0.60:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return {
            "url": url,
            
            "final_decision": {
                "prediction": meta_label,
                "confidence": round(meta_confidence, 4),
                "confidence_level": confidence_level,
                "threat_detected": meta_label != "benign",
                "model_used": "meta_lightgbm_with_smote"
            },
            
            "model_predictions": {
                "lightgbm": {
                    "prediction": results["lgb_result"]["prediction"],
                    "confidence": round(results["lgb_result"]["confidence"], 4),
                    "probabilities": {
                        cls: round(results["lgb_result"]["probabilities"].get(cls, 0.0), 4)
                        for cls in CLASSES
                    },
                    "threat_detected": results["lgb_result"]["prediction"] != "benign"
                },
                "xgboost": {
                    "prediction": results["xgb_result"]["prediction"],
                    "confidence": round(results["xgb_result"]["confidence"], 4),
                    "probabilities": {
                        cls: round(results["xgb_result"]["probabilities"].get(cls, 0.0), 4)
                        for cls in CLASSES
                    },
                    "threat_detected": results["xgb_result"]["prediction"] != "benign"
                },
                "random_forest": {
                    "prediction": results["rf_result"]["prediction"],
                    "confidence": round(results["rf_result"]["confidence"], 4),
                    "probabilities": {
                        cls: round(results["rf_result"]["probabilities"].get(cls, 0.0), 4)
                        for cls in CLASSES
                    },
                    "threat_detected": results["rf_result"]["prediction"] != "benign"
                },
                "meta_ensemble": {
                    "prediction": meta_label,
                    "confidence": round(meta_confidence, 4),
                    "probabilities": {
                        cls: round(float(proba[i]), 4) 
                        for i, cls in enumerate(CLASSES)
                    },
                    "threat_detected": meta_label != "benign"
                }
            },
            
            "agreement_analysis": {
                "all_agree": (
                    results["lgb_result"]["prediction"] == 
                    results["xgb_result"]["prediction"] == 
                    results["rf_result"]["prediction"] == 
                    meta_label
                ),
                "base_models_agree": (
                    results["lgb_result"]["prediction"] == 
                    results["xgb_result"]["prediction"] == 
                    results["rf_result"]["prediction"]
                ),
                "meta_agrees_with_majority": _check_majority_agreement(results, meta_label),
                "disagreement_details": _get_disagreement_details(results, meta_label)
            },
            
            "meta_info": {
                "num_features_used": int(meta_features.shape[1]),
                "base_features": 12,
                "engineered_features": 13,
                "models_used": ["LightGBM", "XGBoost", "RandomForest", "Meta-LightGBM"]
            }
        }

    except Exception as e:
        print(e)
        raise RuntimeError(f"Meta-ensemble prediction failed: {e}")

# def predict_ensemble(url: str):
#     try:
#         results = get_all_model_probs(url)
        
#         # ========== RL AGENT INTERVENTION ==========
        
#         # Extract state features
#         state = extract_rl_state(results, url)
#         state_key = discretize_state(state)
        
#         # RL agent selects action
#         action = rl_agent.select_action(state_key)
#         action_name = rl_agent.action_names[action]
        
#         # Prepare meta features (in case we need them)
#         meta_features_raw = results["combined_vector"].reshape(1, -1)
#         meta_features = engineer_meta_features(meta_features_raw)
        
#         # Execute chosen action
#         final_prediction, final_confidence, final_probabilities = execute_rl_action(
#             action, results, meta_features, meta_model
#         )
        
#         # Store prediction for feedback learning
#         request_id = prediction_buffer.add(
#             state_key=state_key,
#             action=action,
#             prediction=final_prediction,
#             confidence=final_confidence,
#             probabilities=final_probabilities,
#             url=url
#         )
        
#         # ========== END RL LOGIC ==========
        
#         # Determine confidence level
#         if final_confidence > 0.85:
#             confidence_level = "HIGH"
#         elif final_confidence > 0.60:
#             confidence_level = "MEDIUM"
#         else:
#             confidence_level = "LOW"
        
#         return {
#             "url": url,
            
#             "final_decision": {
#                 "prediction": final_prediction,
#                 "confidence": round(final_confidence, 4),
#                 "confidence_level": confidence_level,
#                 "threat_detected": final_prediction != "benign",
#                 "rl_action_used": action_name,
#                 "rl_action_id": action,
#                 "request_id": request_id
#             },
            
#             "model_predictions": {
#                 "lightgbm": {
#                     "prediction": results["lgb_result"]["final_prediction"],
#                     "confidence": round(max(results["lgb_result"]["final_probabilities"].values()), 4),
#                     "probabilities": {
#                         cls: round(results["lgb_result"]["final_probabilities"].get(cls, 0.0), 4)
#                         for cls in CLASSES
#                     },
#                     "threat_detected": results["lgb_result"]["final_prediction"] != "benign"
#                 },
#                 "xgboost": {
#                     "prediction": results["xgb_result"]["final_prediction"],
#                     "confidence": round(max(results["xgb_result"]["final_probabilities"].values()), 4),
#                     "probabilities": {
#                         cls: round(results["xgb_result"]["final_probabilities"].get(cls, 0.0), 4)
#                         for cls in CLASSES
#                     },
#                     "threat_detected": results["xgb_result"]["final_prediction"] != "benign"
#                 },
#                 "random_forest": {
#                     "prediction": results["rf_result"]["final_prediction"],
#                     "confidence": round(max(results["rf_result"]["final_probabilities"].values()), 4),
#                     "probabilities": {
#                         cls: round(results["rf_result"]["final_probabilities"].get(cls, 0.0), 4)
#                         for cls in CLASSES
#                     },
#                     "threat_detected": results["rf_result"]["final_prediction"] != "benign"
#                 },
#                 "rl_ensemble": {
#                     "prediction": final_prediction,
#                     "confidence": round(final_confidence, 4),
#                     "probabilities": {
#                         cls: round(float(final_probabilities.get(cls, 0.0)), 4)
#                         for cls in CLASSES
#                     },
#                     "threat_detected": final_prediction != "benign",
#                     "strategy_used": action_name
#                 }
#             },
            
#             "agreement_analysis": {
#                 "all_agree": (
#                     results["lgb_result"]["final_prediction"] == 
#                     results["xgb_result"]["final_prediction"] == 
#                     results["rf_result"]["final_prediction"]
#                 ),
#                 "base_models_agree": (
#                     results["lgb_result"]["final_prediction"] == 
#                     results["xgb_result"]["final_prediction"] == 
#                     results["rf_result"]["final_prediction"]
#                 ),
#                 "models_in_agreement": 3 - len(set([
#                     results["lgb_result"]["final_prediction"],
#                     results["xgb_result"]["final_prediction"],
#                     results["rf_result"]["final_prediction"]
#                 ]))
#             },
            
#             "rl_info": {
#                 "state_features": {
#                     "all_models_agree": state['all_agree'],
#                     "max_confidence": state['max_conf'],
#                     "confidence_std": state['conf_std'],
#                     "url_entropy": state['url_entropy'],
#                     "brand_impersonation": bool(state['brand_impersonation']),
#                     "typosquatting": bool(state['is_typosquatting'])
#                 },
#                 "discretized_state": str(state_key),
#                 "action_selected": action,
#                 "action_name": action_name,
#                 "epsilon": rl_agent.epsilon,
#                 "q_values": {
#                     rl_agent.action_names[i]: float(rl_agent.q_table[state_key][i])
#                     for i in range(rl_agent.n_actions)
#                 }
#             },
            
#             "meta_info": {
#                 "num_features_used": int(meta_features.shape[1]),
#                 "base_features": 12,
#                 "engineered_features": 13,
#                 "models_used": ["LightGBM", "XGBoost", "RandomForest", "Meta-LightGBM", "RL-Controller"]
#             }
#         }

#     except Exception as e:
#         print(f"⚠️  Error in predict_ensemble: {e}")
#         import traceback
#         traceback.print_exc()
#         raise RuntimeError(f"Meta-ensemble prediction failed: {e}")

def _check_majority_agreement(results, meta_label):
    """Check if meta-learner agrees with majority of base models"""
    predictions = [
        results["lgb_result"]["prediction"],
        results["xgb_result"]["prediction"],
        results["rf_result"]["prediction"]
    ]
    from collections import Counter
    most_common = Counter(predictions).most_common(1)[0][0]
    return meta_label == most_common


def _get_disagreement_details(results, meta_label):
    """Get details about which models disagree"""
    base_predictions = {
        "lightgbm": results["lgb_result"]["prediction"],
        "xgboost": results["xgb_result"]["prediction"],
        "random_forest": results["rf_result"]["prediction"]
    }
    
    disagreements = []
    for model, pred in base_predictions.items():
        if pred != meta_label:
            disagreements.append({
                "model": model,
                "predicted": pred,
                "meta_predicted": meta_label
            })
    
    return disagreements if disagreements else None