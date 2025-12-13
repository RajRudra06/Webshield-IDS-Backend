# import pandas as pd
# import joblib
# from pathlib import Path
# from ..helperScripts.feature_extraction import extract_features_enhanced
# from ..helperScripts.typoSquattingFunction import apply_typosquatting_heuristic 

# BASE_DIR = Path(__file__).resolve().parents[3]
# rf_path = BASE_DIR / "models" / "716k typosquatting" / "rf classifier v_3.pkl"

# artifact = joblib.load(rf_path)

# model = artifact['model']
# features = artifact['feature_names']

# def process_url_with_heuristic_rf(url):

#     if not url.startswith(('http://', 'https://')):
#         url = 'https://' + url

#     X_dict = extract_features_enhanced(url)
#     X = pd.DataFrame([X_dict])

#     for col in features:
#         if col not in X.columns:
#             X[col] = 0
#     X = X[features]

#     model_pred = model.predict(X)[0]         
#     model_proba = model.predict_proba(X)[0]   
#     classes = model.classes_                  
#     prob_dict = {cls: float(prob) for cls, prob in zip(classes, model_proba)}

#     final_pred, final_proba, reason = apply_typosquatting_heuristic(
#         url, model_pred, prob_dict
#     )

#     return {
#         'url': url,
#         'model_prediction': model_pred,
#         'model_probabilities': prob_dict,
#         'final_prediction': final_pred,
#         'final_probabilities': final_proba,
#         'detection_reason': reason,
#         'heuristic_applied': reason != "model_decision"
#     }

import pandas as pd
from pathlib import Path
import joblib
from ..helperScripts.feature_extraction import extract_features_enhanced
from ..helperScripts.typoSquattingFunction import apply_typosquatting_heuristic

_BASE_DIR = Path(__file__).resolve().parents[3]
_RF_PATH = _BASE_DIR / "models" / "716k typosquatting" / "rf classifier v_3.pkl"

_rf_model = None
_rf_features = None


def _load_rf():
    global _rf_model, _rf_features
    if _rf_model is None:
        artifact = joblib.load(_RF_PATH)
        _rf_model = artifact["model"]
        _rf_features = artifact["feature_names"]
    return _rf_model, _rf_features


def process_url_with_heuristic_rf(url: str):
    model, features = _load_rf()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    X_dict = extract_features_enhanced(url)
    X = pd.DataFrame([X_dict])

    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]

    model_pred = model.predict(X)[0]
    model_proba = model.predict_proba(X)[0]
    classes = model.classes_
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, model_proba)}

    final_pred, final_proba, reason = apply_typosquatting_heuristic(
        url, model_pred, prob_dict
    )

    return {
        "url": url,
        "model_prediction": model_pred,
        "model_probabilities": prob_dict,
        "final_prediction": final_pred,
        "final_probabilities": final_proba,
        "detection_reason": reason,
        "heuristic_applied": reason != "model_decision",
    }
