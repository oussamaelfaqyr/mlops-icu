from xgboost import XGBClassifier


def get_model(params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,  # For reproducibility
        }
    return XGBClassifier(**params)
