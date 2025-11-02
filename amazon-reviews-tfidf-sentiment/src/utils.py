from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, 
    auc, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
    )
import matplotlib.pyplot as plt
import numpy as np

### Summarize Results Function
def summarize(name, y_true, y_pred, y_proba):
    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc  = roc_auc_score(y_true, y_proba)
    ap   = average_precision_score(y_true, y_proba)  # PR-AUC
    print(f"\n== {name} ==")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {ap:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

### Function to Generate Confusion Matrix
def cm(y_test, y_preds, model_names=None, plot_name=None):
    if not isinstance(y_preds, list):
        y_preds = [y_preds]

    n_models = len(y_preds)
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]

    fig, axs = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axs = [axs] 

    for n, (preds, name) in enumerate(zip(y_preds, model_names)):

        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Neg","Pos"] if len(labels)==2 else labels)
        disp.plot(ax=axs[n], cmap="Blues", values_format="d", colorbar=False)
        axs[n].set_title(f"Confusion Matrix - {name}", fontsize=13)
        axs[n].set_xlabel("Predicted_label", fontsize=12)
        axs[n].set_ylabel("True Label", fontsize=12)

    if plot_name:
        plt.figsave(plot_name, dpi=200, bbox_inches="tight")
    plt.tight_layout() 
    plt.show()


### Function to Generate ROC and Precision-Recall Plots
def ROC_AUC_PR_AP(y_test, y_probas, model_names=None, plot_name=None):
    
    """
    y_test : array-like,
    y_probas : list, [proba1, proba2, ...] (shape = (n_samples,))
    model_names : list of str, if None (Model 1, Model 2, ...)
    """

    if not isinstance(y_probas, list):
        y_probas = [y_probas]
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(y_probas))]

    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    dic = {}

    for proba, name in zip(y_probas, model_names):

        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)

        axs[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
        axs[1].plot(rec, prec, label=f"{name} (AP = {auc:.2f})")

        dic[name] = {"auc": auc_score, "ap": ap}

    axs[0].plot([0,1],[0,1],"--", lw=1)
    axs[0].set_xlabel("False Positive Rate", fontsize=13)
    axs[0].set_ylabel("True Positive Rate", fontsize=13)
    axs[0].set_title("ROC Curve", fontsize=15)
    axs[0].legend(loc='lower right')

    axs[1].plot(prec, rec, label=f"LR (AP = {ap:.2f})")
    axs[1].set_xlabel("Recall", fontsize=13)
    axs[1].set_ylabel("Precision", fontsize=13); 
    axs[1].set_title("Precision-Recall Curve", fontsize=15)
    axs[1].legend(loc='lower left')

    if plot_name:
        plt.figsave(plot_name, dpi=200, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    df_scores = pd.DataFrame.from_dict(dic, orient="index").reset_index().rename(columns={"index": "model"})

    return df_scores


### Find Best Threshold by F1
def best_threshold_by_f1(y_true, y_proba):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = (2*prec*rec)/(prec+rec+1e-12)
    i = np.nanargmax(f1[:-1])
    return float(thr[i]), float(f1[i]), float(prec[i]), float(rec[i])
