import numpy as np
from collections import Counter


def execute_rl_action(action, results, meta_features, meta_model):
    
    CLASSES = ['benign', 'defacement', 'malware', 'phishing']
    
    if action == 0:  # trust_consensus_override
        return action_consensus_override(results)
    
    elif action == 1:  # trust_meta_model
        return action_meta_model(meta_features, meta_model, CLASSES)
    
    elif action == 2:  # trust_highest_confidence_base
        return action_highest_confidence(results)
    
    elif action == 3:  # weighted_average
        return action_weighted_average(results, CLASSES)
    
    elif action == 4:  # unanimous_high_conf_only
        return action_unanimous_high_conf(results, meta_features, meta_model, CLASSES)
    
    else:
        # Fallback to meta model
        return action_meta_model(meta_features, meta_model, CLASSES)


def action_consensus_override(results):
    """Action 0: Use consensus if all base models agree"""
    base_preds = [
        results["lgb_result"]["final_prediction"],
        results["xgb_result"]["final_prediction"],
        results["rf_result"]["final_prediction"]
    ]
    
    if len(set(base_preds)) == 1:  # All agree
        unanimous_label = base_preds[0]
        unanimous_conf = max(
            max(results["lgb_result"]["final_probabilities"].values()),
            max(results["xgb_result"]["final_probabilities"].values()),
            max(results["rf_result"]["final_probabilities"].values())
        )
        
        # Use average of probabilities from all models
        all_probs = {}
        for cls in results["lgb_result"]["final_probabilities"].keys():
            avg_prob = np.mean([
                results["lgb_result"]["final_probabilities"].get(cls, 0),
                results["xgb_result"]["final_probabilities"].get(cls, 0),
                results["rf_result"]["final_probabilities"].get(cls, 0)
            ])
            all_probs[cls] = float(avg_prob)
        
        return unanimous_label, unanimous_conf, all_probs
    else:
        # No consensus, use highest confidence
        return action_highest_confidence(results)


def action_meta_model(meta_features, meta_model, CLASSES):
    """Action 1: Use meta-model prediction"""
    proba = meta_model.predict_proba(meta_features)[0]
    idx = int(np.argmax(proba))
    meta_label = CLASSES[idx]
    meta_confidence = float(proba[idx])
    
    prob_dict = {cls: float(proba[i]) for i, cls in enumerate(CLASSES)}
    
    return meta_label, meta_confidence, prob_dict


def action_highest_confidence(results):
    """Action 2: Choose base model with highest confidence"""
    base_results = [
        ("lightgbm", results["lgb_result"]),
        ("xgboost", results["xgb_result"]),
        ("random_forest", results["rf_result"])
    ]
    
    best_model, best_result = max(
        base_results,
        key=lambda x: max(x[1]["final_probabilities"].values())
    )
    
    prediction = best_result["final_prediction"]
    confidence = max(best_result["final_probabilities"].values())
    probabilities = best_result["final_probabilities"]
    
    return prediction, confidence, probabilities


def action_weighted_average(results, CLASSES):
    """Action 3: Weighted average based on confidence"""
    # Get confidences as weights
    lgb_conf = max(results["lgb_result"]["final_probabilities"].values())
    xgb_conf = max(results["xgb_result"]["final_probabilities"].values())
    rf_conf = max(results["rf_result"]["final_probabilities"].values())
    
    total_conf = lgb_conf + xgb_conf + rf_conf
    
    # Normalize weights
    w_lgb = lgb_conf / total_conf
    w_xgb = xgb_conf / total_conf
    w_rf = rf_conf / total_conf
    
    # Weighted average of probabilities
    weighted_probs = {}
    for cls in CLASSES:
        weighted_probs[cls] = (
            w_lgb * results["lgb_result"]["final_probabilities"].get(cls, 0) +
            w_xgb * results["xgb_result"]["final_probabilities"].get(cls, 0) +
            w_rf * results["rf_result"]["final_probabilities"].get(cls, 0)
        )
    
    # Get prediction
    prediction = max(weighted_probs, key=weighted_probs.get)
    confidence = weighted_probs[prediction]
    
    return prediction, confidence, weighted_probs


def action_unanimous_high_conf(results, meta_features, meta_model, CLASSES):
    """Action 4: Only trust if all models agree AND high confidence"""
    base_preds = [
        results["lgb_result"]["final_prediction"],
        results["xgb_result"]["final_prediction"],
        results["rf_result"]["final_prediction"]
    ]
    
    base_confs = [
        max(results["lgb_result"]["final_probabilities"].values()),
        max(results["xgb_result"]["final_probabilities"].values()),
        max(results["rf_result"]["final_probabilities"].values())
    ]
    
    # Check if unanimous AND all high confidence (>0.75)
    if len(set(base_preds)) == 1 and min(base_confs) > 0.75:
        return action_consensus_override(results)
    else:
        # Fall back to meta model
        return action_meta_model(meta_features, meta_model, CLASSES)