# Telco Customer Churn Prediction


## Goal
Develop machine learning models to predict whether a telecom customer will churn based on service usage and demographic data.

---

## Data
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Customer demographics: gender, age range, partner, dependents
- Customer account: tenure, contract type, payment method, billing, charges
- Services subscribed: phone, multiple lines, internet, online security, backup, device protection, tech support, streaming TV/movies
- Churn: whether the customer left in the last month

---

## Methods
- **EDA:** 
  - Churn Distribution: About 26% of customers left (imbalanced target).
  - Customers with <10 months tenure tend to churn more.
  - Higher monthly charges → higher churn.
  - Total charges show no clear churn pattern.
  - Month-to-month contracts have the highest churn rate.
  - Paperless billing and manual payments increase churn.
  - Fiber optic users churn more than others.
  - Lack of Online Security or Tech Support → higher churn.

- **Results Comparison**
  - For Random Forest and XGBoost, Recall values with GridSearch best parameters were included.
<div align="center">

| Model               | Accuracy  | Precision (Churn)  | Recall (Churn)   | F1 (Churn)  |
|---------------------|:---------:|:------------------:|:----------------:|:-----------:|
| Logistic Regression | 0.8006    | 0.6467             | 0.5481           | 0.5933      |
| LogReg (balanced)   | 0.7381    | 0.5043             | 0.7914           | 0.6160      |
| LogReg + SMOTE      | 0.7324    | 0.4969             | 0.6444           | 0.5611      |
| Random Forest       | 0.7793    | 0.5763             | 0.6364 -> 0.7139 | 0.6048      |
| XGBoost             | 0.7679    | 0.5575             | 0.6096 -> 0.6497 | 0.5824      |
| LightGBM            | 0.7800    | 0.5773             | 0.6390           | 0.6066      |

</div>
<!-- 0.8006    0.6467    0.5481    0.5933 (LR)
0.7381    0.5043    0.7914    0.6160 (LR-bal)
0.7324    0.4969    0.6444    0.5611 (SMOTE)
0.7793    0.5763    0.6364    0.6048 (RF)
0.7679    0.5575    0.6096    0.5824 (XGB)
0.7800    0.5773    0.6390    0.6066 (LGBM)
0.5483    0.7139    0.6202 (RF-B)
0.5283    0.6497    0.5827 (XGB-B) -->

- **Logistic Regression (baseline):**
  - Strong overall, but poor recall on minority churn class.  

- **Logistic Regression (balanced weights):**
  - Improved churn recall to 0.79, but precision decreased.  

- **Logistic Regression + SMOTE**
  - Recall improved compared to baseline LogReg, but precision remains low.  

- **Random Forest**
  - Showed balanced performance (Precision 0.58, Recall 0.64).
  - With GridSearch, recall increased to 0.71 (Precision 0.55, not shown in the table).  

- **XGBoost**
  - Churn recall: 0.61 — stronger than Logistic Regression baseline.
  - With GridSearch, recall increased to 0.65.  

- **LightGBM**
  - Recall (0.64) is similar to LogReg + SMOTE model, but Precision is higher.



---

## Results Figures

### Confusion Matrices
- Logistic Regression  (Baseline & Balanced Weights & SMOTE)
  <img src="reports/figures/baseline/heatmap_baseline_models.png" width="400"> 

- Tree-based Models (RF & XGB & LGBM)
  <img src="reports/figures/tree-based/tree_based_combined.png" width="400">  

- Tree-based Models (RF & XGB with best params)  
  <img src="reports/figures/tree-based/combined_best_params.png" width="400">  

---

## Summary from the Results

- **High recall options:** LogReg (balanced) and RandomForest (with GridSearch) yield higher recall (0.79 and 0.71), making them more suitable if capturing most churn cases is critical, though at the cost of precision and accuracy.

- **Modeling insight:** Tree-based methods (Random Forest, XGBoost, LightGBM) consistently provide stronger recall–precision balance for churn prediction.

---

## Key Insights from Preprocessing and Modeling
*Figures showing feature importances coefficients are in reports/figures/baseline (or tree-based)*

- **Tenure** is a strong predictor of churn.
  - Customers with less than 10 months tenure show the highest churn rate.
  - Noticeable churn differences occur around the 10- and 20-month marks.
  - Logistic Regression (both regular and balanced) shows negative coefficients (-0.6 to -1.24), indicating that longer tenure reduces churn likelihood.
  - Even after applying SMOTE, tenure remains a major influencing factor.

- **Contract Length:**
  - Two-year contracts reduce churn more effectively than one-year contracts.
  - Random Forest results further emphasize the importance of total charges.

- **Total Charges** show a potentially strong relationship with churn, though this was not clearly visible during EDA. Further analysis is recommended.

- **Service Subscriptions** (e.g., Online Security, Tech Support) appeared important in EDA, but their impact on model predictions is relatively small, suggesting the need for further inference analysis.

- **Fiber Optic Internet** and **Payment Method** may also influence churn, as suggested by SMOTE model coefficients.

## Next Steps:

Examine the effects of Service Substription, Monthly/Total Charges and Fiber Optic Internet in more detail.