from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    objective="binary",
    random_state=42,
    n_jobs=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    device='gpu',
    scale_pos_weight=scale_pos
)

param_grid = {
    'n_estimators': [300, 500],
    'num_leaves': [25, 31, 40],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7]
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=-1
)

grid_search.fit(X_train, y_train)