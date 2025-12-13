# import pandas as pd
# import numpy as np
# import joblib
# from pathlib import Path
# from sklearn.preprocessing import LabelEncoder
# from ..helperScripts.feature_extraction import extract_features_enhanced
# from ..helperScripts.typoSquattingFunction import apply_typosquatting_heuristic 


# BASE_DIR = Path(__file__).resolve().parents[3]
# xg_path = BASE_DIR / "models" / "716k typosquatting" / "xgboost classifier v_3.pkl"

# artifact = joblib.load(xg_path)
# model = artifact['model']
# features = artifact['feature_names']



# encoder_path = ""
# try:
#     le = joblib.load(encoder_path)
# except:
#     le = LabelEncoder()
#     le.classes_ = np.array(['benign', 'defacement', 'malware', 'phishing'])

# def process_url_with_heuristic_xgboost(url):

#     if not url.startswith(('http://', 'https://')):
#         url = 'https://' + url

#     X_dict = extract_features_enhanced(url)
#     X = pd.DataFrame([X_dict])

#     for col in features:
#         if col not in X.columns:
#             X[col] = 0
#     X = X[features]

#     proba = model.predict_proba(X)[0]             
#     pred_numeric = int(np.argmax(proba))         
#     model_pred = le.inverse_transform([pred_numeric])[0] 

#     prob_dict = {le.classes_[i]: round(float(proba[i]), 4)
#                  for i in range(len(le.classes_))}

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
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder
from ..helperScripts.feature_extraction import extract_features_enhanced
from ..helperScripts.typoSquattingFunction import apply_typosquatting_heuristic

_BASE_DIR = Path(__file__).resolve().parents[3]
_XGB_PATH = _BASE_DIR / "models" / "716k typosquatting" / "xgboost classifier v_3.pkl"

_xgb_model = None
_xgb_features = None
_label_encoder = None


def _load_xgb():
    global _xgb_model, _xgb_features, _label_encoder
    if _xgb_model is None:
        artifact = joblib.load(_XGB_PATH)
        _xgb_model = artifact["model"]
        _xgb_features = artifact["feature_names"]

        le = LabelEncoder()
        le.classes_ = np.array(["benign", "defacement", "malware", "phishing"])
        _label_encoder = le

    return _xgb_model, _xgb_features, _label_encoder


def process_url_with_heuristic_xgboost(url: str):
    model, features, le = _load_xgb()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    X_dict = extract_features_enhanced(url)
    X = pd.DataFrame([X_dict])

    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]

    proba = model.predict_proba(X)[0]
    pred_numeric = int(np.argmax(proba))
    model_pred = le.inverse_transform([pred_numeric])[0]

    prob_dict = {
        le.classes_[i]: float(proba[i])
        for i in range(len(le.classes_))
    }

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
