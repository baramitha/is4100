# SCRUM-7: Model Development
# Assignee: Naura
# Sprint: 2
# Epic: SCRUM-7 Model Development
 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")
 
 
# ============================================================
# DEFINITION OF DONE (SCRUM-13):
# Model is DONE when F1 Score > 0.75 on test set
# ============================================================
F1_THRESHOLD = 0.75
 
 
def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    SCRUM-7: Train Logistic Regression model for customer churn prediction.
    """
    print("\n========== LOGISTIC REGRESSION ==========")
 
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"  # handles class imbalance
    )
    model.fit(X_train, y_train)
    print("✅ Logistic Regression model trained")
 
    return model
 
 
def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    SCRUM-7: Train Random Forest model for customer churn prediction.
    """
    print("\n========== RANDOM FOREST ==========")
 
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"  # handles class imbalance
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest model trained")
 
    return model
 
 
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    SCRUM-7: Evaluate model against Definition of Done (F1 > 0.75).
    """
    print(f"\n========== EVALUATION: {model_name.upper()} ==========")
 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
 
    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
 
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
 
    # Definition of Done check
    print(f"\n🏁 Definition of Done Check (F1 > {F1_THRESHOLD}):")
    if f1 >= F1_THRESHOLD:
        print(f"   ✅ PASSED — F1={f1:.4f} meets threshold of {F1_THRESHOLD}")
    else:
        print(f"   ❌ FAILED — F1={f1:.4f} below threshold of {F1_THRESHOLD}")
        print(f"   ⚠️  Action: Tune hyperparameters or try feature selection")
 
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔲 Confusion Matrix:")
    print(f"   True Negative  (TN): {cm[0][0]}  — correctly predicted 'No Churn'")
    print(f"   False Positive (FP): {cm[0][1]}  — predicted churn but didn't")
    print(f"   False Negative (FN): {cm[1][0]}  — missed actual churners")
    print(f"   True Positive  (TP): {cm[1][1]}  — correctly predicted 'Churn'")
 
    # Full classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
 
    results = {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "dod_passed": f1 >= F1_THRESHOLD,
        "confusion_matrix": cm.tolist()
    }
 
    print("=" * 50)
    return results
 
 
def compare_models(results_lr: dict, results_rf: dict):
    """
    SCRUM-7: Compare Logistic Regression vs Random Forest.
    """
    print("\n========== MODEL COMPARISON ==========")
    print(f"\n{'Metric':<20} {'Logistic Regression':<25} {'Random Forest':<25}")
    print("-" * 70)
    for metric in ["accuracy", "f1_score", "roc_auc"]:
        lr_val = results_lr[metric]
        rf_val = results_rf[metric]
        winner = "← BETTER" if lr_val > rf_val else ("← BETTER" if rf_val > lr_val else "TIE")
        rf_note = "← BETTER" if rf_val > lr_val else ""
        print(f"{metric:<20} {lr_val:<25} {rf_val:<15} {rf_note}")
 
    # Recommend best model
    if results_rf["f1_score"] >= results_lr["f1_score"]:
        print(f"\n🏆 Recommended Model: Random Forest (F1={results_rf['f1_score']})")
    else:
        print(f"\n🏆 Recommended Model: Logistic Regression (F1={results_lr['f1_score']})")
 
    print("=" * 50)
 
 
def get_feature_importance(model, feature_names: list, model_name: str):
    """
    SCRUM-7: Show top 10 most important features for churn prediction.
    """
    print(f"\n🔍 Top 10 Feature Importances ({model_name}):")
 
    if hasattr(model, "feature_importances_"):
        # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression
        importances = abs(model.coef_[0])
    else:
        print("Feature importance not available for this model")
        return
 
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10)
 
    for i, row in feat_df.iterrows():
        bar = "█" * int(row["Importance"] * 100)
        print(f"   {row['Feature']:<35} {row['Importance']:.4f} {bar}")
 
 
if __name__ == "__main__":
    from data_exploration import load_data
    from data_cleaning import clean_data
    from feature_engineering import engineer_features
    from train_test_split import split_data, validate_split
 
    # Full pipeline
    print("🚀 Running IS4100 ML Pipeline — Customer Churn Prediction")
    print("=" * 60)
 
    df = load_data("data/raw/churn.csv")
    df = clean_data(df)
    X, y, feature_names, scaler = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    validate_split(X_train, X_test, y_train, y_test)
 
    # Train both models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
 
    # Evaluate both models
    results_lr = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")
 
    # Compare
    compare_models(results_lr, results_rf)
 
    # Feature importance
    get_feature_importance(lr_model, feature_names, "Logistic Regression")
    get_feature_importance(rf_model, feature_names, "Random Forest")
 
    print("\n✅ Pipeline complete!")
