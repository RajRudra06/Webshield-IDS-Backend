import numpy as np
from ..helperScripts.feature_extraction import extract_features_enhanced


def extract_rl_state(results, url):
   
    # Extract base model predictions and confidences
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
    
    # Extract URL features
    try:
        url_features = extract_features_enhanced(url)
    except:
        url_features = {
            'url_entropy': 0,
            'brand_impersonation': 0,
            'is_typosquatting': 0,
            'is_suspicious_tld': 0
        }
    
    # Compute state features
    state = {
        'all_agree': int(len(set(base_preds)) == 1),
        'agreement_count': 3 - len(set(base_preds)),
        'max_conf': max(base_confs),
        'min_conf': min(base_confs),
        'conf_std': float(np.std(base_confs)),
        'url_entropy': url_features.get('url_entropy', 0),
        'brand_impersonation': url_features.get('brand_impersonation', 0),
        'is_typosquatting': url_features.get('is_typosquatting', 0),
        'suspicious_tld': url_features.get('is_suspicious_tld', 0)
    }
    
    return state


def discretize_state(state):
    
    # Discretize continuous values into bins
    
    # Agreement (already discrete)
    all_agree = state['all_agree']
    agreement_count = state['agreement_count']
    
    # Confidence bins: LOW (0-0.6), MED (0.6-0.85), HIGH (0.85-1.0)
    def conf_bin(conf):
        if conf < 0.6:
            return 0  # LOW
        elif conf < 0.85:
            return 1  # MEDIUM
        else:
            return 2  # HIGH
    
    max_conf_bin = conf_bin(state['max_conf'])
    min_conf_bin = conf_bin(state['min_conf'])
    
    # Confidence std bins: LOW (0-0.1), MED (0.1-0.2), HIGH (0.2+)
    conf_std = state['conf_std']
    if conf_std < 0.1:
        conf_std_bin = 0
    elif conf_std < 0.2:
        conf_std_bin = 1
    else:
        conf_std_bin = 2
    
    # URL entropy bins: LOW (0-3), MED (3-4), HIGH (4+)
    url_entropy = state['url_entropy']
    if url_entropy < 3.0:
        entropy_bin = 0
    elif url_entropy < 4.0:
        entropy_bin = 1
    else:
        entropy_bin = 2
    
    # Binary features (already discrete)
    brand_imp = state['brand_impersonation']
    typosquat = state['is_typosquatting']
    susp_tld = state['suspicious_tld']
    
    # Create hashable state key
    state_key = (
        all_agree,
        agreement_count,
        max_conf_bin,
        min_conf_bin,
        conf_std_bin,
        entropy_bin,
        brand_imp,
        typosquat,
        susp_tld
    )
    
    return state_key