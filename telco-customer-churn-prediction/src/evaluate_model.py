from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            precision_recall_fscore_support, roc_auc_score)
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, prefix, X_train, X_test, y_train, y_test, plot_cm=False):
    
    estimate = clone(model)
    estimate.fit(X_train, y_train)
    y_pred = estimate.predict(X_test)

    try:
        y_score = estimate.predict_prob(X_test)[:, 1] # get only churn
        auc = roc_auc_score(y_test, y_score)
    except:
        y_score = None
        auc = None

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    print(f"{prefix} Results:\n")
    print(f"\n[{prefix}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}", (f"  ROC-AUC={auc:.4f}" if auc is not None else ""))

    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)


    if plot_cm:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"], 
                    annot_kws={'size': 14})
        plt.title(f"{prefix} - Confusion Matrix Plot", fontsize=16)
        plt.ylabel("True", fontsize=14); plt.xlabel("Predicted", fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        plt.savefig(f"figures/{prefix}.png", dpi=200, bbox_inches='tight')
        plt.show()

    return