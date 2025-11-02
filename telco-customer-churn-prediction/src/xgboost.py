from xgboost import XGBClassifier


pos = y_train.sum()
neg = len(y_train) - pos
scale_pos = neg / pos if pos > 0 else 1.0  # For XGBoost

def build_xgb():
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        n_jobs=-1,
        random_state=42,
        tree_method="gpu_hist",
        eval_metric="logloss",
        scale_pos_weight=scale_pos
    )


xgb = build_xgb()