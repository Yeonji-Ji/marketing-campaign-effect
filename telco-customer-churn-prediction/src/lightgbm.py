from lightgbm import LGBMClassifier
def build_lgbm():
    return LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.08,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        n_jobs=-1,
        device="gpu",
        random_state=42,
        eval_metric="binary_logloss",
        scale_pos_weight=scale_pos
    )


lgbm = build_lgbm()