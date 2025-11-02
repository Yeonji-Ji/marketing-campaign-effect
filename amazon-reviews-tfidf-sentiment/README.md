# Amazon Reviews NLP Sentiment Classification<br> with TF-IDF and ML baselines

## 1. Introduction
This project aims to classify Amazon product reviews into **positive** and **negative** sentiments using machine learning models.  
The goal is to demonstrate practical NLP skills (data preprocessing, feature engineering with TF-IDF, baseline modeling, and evaluation) in a real-world context.

## 2. Exploratory Data Analysis (EDA)
- **Label distribution**: Balanced dataset between positive and negative reviews.  
- **Text length**: Most reviews are under 200 words.  
- **Frequent words**: Common words include *product*, *time*, *good*, *bad*. Stopwords were removed for modeling.

## 3. Models
We trained and compared several classifiers using a **TF-IDF representation**:
- **Logistic Regression**
- **Multinomial/Complement Naive Bayes**
- **Linear Support Vector Classifier (LinearSVC)**
- **XGBoost (tree-based)**

Hyperparameter tuning was applied (e.g., `ngram_range`, `C` for SVM, `alpha` for NB).

## 4. Results
| Model                    | Accuracy | Precision | Recall | F1    | AUC   | AP    |
|--------------------------|----------|-----------|--------|-------|-------|-------|
| Logistic Regression      | 0.889    | 0.887     | 0.895  | 0.891 | 0.956 | 0.956 |
| Multinomial Naive Bayes  | 0.858    | 0.859     | 0.861  | 0.860 | 0.935 | 0.935 |
| Complement Naive Bayes   | 0.858    | 0.862     | 0.856  | 0.859 | 0.935 | 0.935 |
| LinearSVC                | 0.886    | 0.884     | 0.893  | 0.888 | 0.955 | 0.954 |
| LinearSVD (GridSearch)   | 0.886    | 0.884     | 0.893  | 0.888 | 0.957 | 0.956 |
| XGBoost (TF-IDF)         | 0.589    | 0.757     | 0.281  | 0.410 | 0.597 | 0.584 |
| XGBoost (RandomizeSearch)| 0.602    | 0.754     | 0.320  | 0.449 | 0.655 | 0.637 |


### Key Insights
- **Linear models (SVC, Logistic Regression)** clearly outperform XGBoost in this setting.  
- The reason: **TF-IDF produces a very high-dimensional and sparse feature space**, where linear classifiers are more effective.  
- **Tree-based models like XGBoost** shine on low-dimensional, tabular data, but struggle with sparse text representations.  
- This demonstrates the importance of **choosing algorithms that match the data characteristics**, not assuming one model fits all.

## 5. Conclusion
- Logistic Regression and LinearSVC provide strong, reliable baselines for text classification tasks.  
- XGBoost’s weaker performance highlights that **complex models are not always better** — understanding the data is crucial.  
- Future work could explore deep learning approaches (e.g., LSTMs, Transformers) for potential performance gains.

---

