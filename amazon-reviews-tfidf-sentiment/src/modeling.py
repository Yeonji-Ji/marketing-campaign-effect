
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

### TF-IDF & Logistic Regression
logrep_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
    max_features = 10000,
    ngram_range = (1,2),
    stop_words = "english"
    )), 
    ("clf", LogisticRegression(solver="liblinear",
        class_weight=None, 
        max_iter=1000,
        random_state=42
    ))
])

### TF-IDF & Naive-Bayes

from sklearn.naive_bayes import ComplementNB, MultinomialNB

nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        stop_words="english"
    )),
    ("clf", ComplementNB(alpha=0.5))
])

nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        stop_words="english"
    )),
    ("clf", MultinomialNB())
])

### TF-IDF & LinearSVC

from sklearn.svm import LinearSVC

svm_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english"
    )),
    ("clf", LinearSVC(
        C=1.0,
        class_weight=None,
        max_iter=2000,
        random_state=42
    ))
])

### TF-IDF & LinearSVC (GridSearchCV)

from sklearn.model_selection import GridSearchCV

param_grid = {
    "tfidf__max_features": [10000, 20000],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.5, 1.0, 2.0]
}

svm_search = GridSearchCV(
    estimator=svm_pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

### TF-IDF & XGBClassifier

from xgboost import XGBClassifier

xgb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english"
    )),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        n_jobs=-1,
        random_state=42,
        tree_method="gpu_hist",
        eval_metric="logloss"
    ))
])
