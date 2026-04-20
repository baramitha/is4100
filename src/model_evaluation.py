# SCRUM-23: Model Evaluation Metrics
# Assignee: Naura
# Sprint: 2
# Epic: SCRUM-8 Evaluation & Closeout
 
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
 
 
# Definition of Done: F1 > 0.75
F1_THRESHOLD = 0.75
 
 
def evaluate_full(model, X_test, y_test, model_name: str) -> dict:
    """
    SCRUM-23: Full model evaluation with all metrics.
    """
    print(f"\n========== SCRUM-23: MODEL EVALUATION — {model_name.upper()} ==========")
 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
 
    # Core metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)
 
    print(f"\n📊 Core Metrics:")
    print(f"   {'Accuracy':<20} {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"   {'Precision':<20} {precision:.4f}  — of predicted churners, how many actually churned")
    print(f"   {'Recall':<20} {recall:.4f}  — of actual churners, how many did we catch")
    print(f"   {'F1 Score':<20} {f1:.4f}  — balance of precision and recall")
    print(f"   {'ROC-AUC':<20} {roc_auc:.4f}  — overall discrimination ability")
 
    # Definition of Done check
    print(f"\n🏁 Definition of Done (F1 > {F1_THRESHOLD}):")
    if f1 >= F1_THRESHOLD:
        print(f"   ✅ PASSED — F1={f1:.4f}")
    else:
        print(f"   ❌ FAILED — F1={f1:.4f} (need {F1_THRESHOLD - f1:.4f} more)")
 
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n🔲 Confusion Matrix:")
    print(f"   ┌─────────────────────┬─────────────────────┐")
    print(f"   │ True Negative: {tn:<5} │ False Positive: {fp:<5}│")
    print(f"   │ (Correct No Churn)  │ (Wrong — said churn)│")
    print(f"   ├─────────────────────┼─────────────────────┤")
    print(f"   │ False Negative: {fn:<4} │ True Positive: {tp:<5}│")
    print(f"   │ (Missed churners)   │ (Correct Churn) ✅  │")
    print(f"   └─────────────────────┴─────────────────────┘")
 
    # Business interpretation
    print(f"\n💼 Business Interpretation:")
    print(f"   Out of {len(y_test)} customers:")
    print(f"   ✅ Correctly identified {tp} churners → retention team can act")
    print(f"   ⚠️  Missed {fn} churners → these customers may leave undetected")
    print(f"   ⚠️  False alarms on {fp} customers → wasted retention effort")
 
    # Full classification report
    print(f"\n📋 Full Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Churn", "Churn"]))
 
    results = {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "dod_passed": f1 >= F1_THRESHOLD,
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp)
        }
    }
 
    print("=" * 60)
    return results
 
 
def cross_validate_model(model, X, y, cv: int = 5) -> dict:
    """
    SCRUM-23: Run k-fold cross validation for robust evaluation.
    """
    from sklearn.model_selection import cross_val_score
 
    print(f"\n🔄 {cv}-Fold Cross Validation:")
 
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
 
    print(f"   F1 per fold:  {[round(s, 4) for s in f1_scores]}")
    print(f"   F1 mean:      {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"   Accuracy mean:{acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
 
    if f1_scores.mean() >= F1_THRESHOLD:
        print(f"   ✅ Cross-validated F1 meets DoD threshold of {F1_THRESHOLD}")
    else:
        print(f"   ❌ Cross-validated F1 below DoD threshold of {F1_THRESHOLD}")
 
    return {
        "cv_f1_mean": round(f1_scores.mean(), 4),
        "cv_f1_std": round(f1_scores.std(), 4),
        "cv_accuracy_mean": round(acc_scores.mean(), 4)
    }
 
 
if __name__ == "__main__":
    from src.data_exploration import load_data
    from src.data_cleaning import clean_data
    from src.feature_engineering import engineer_features
    from src.train_test_split import split_data
    from src.model import train_random_forest, train_logistic_regression
 
    df = load_data("data/raw/churn.csv")
    df = clean_data(df)
    X, y, feature_names, scaler = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    rf_model = train_random_forest(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
 
    results_rf = evaluate_full(rf_model, X_test, y_test, "Random Forest")
    results_lr = evaluate_full(lr_model, X_test, y_test, "Logistic Regression")
 
    cross_validate_model(rf_model, X, y)
    cross_validate_model(lr_model, X, y)
