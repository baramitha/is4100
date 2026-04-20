# SCRUM-24: Model Comparison Report
# Assignee: Naura
# Sprint: 2
# Epic: SCRUM-8 Evaluation & Closeout

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


# Definition of Done: F1 > 0.75
F1_THRESHOLD = 0.75


def generate_comparison_report(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    SCRUM-24: Generate a full comparison report for all trained models.
    models = {"Logistic Regression": lr_model, "Random Forest": rf_model}
    """
    print("\n========== SCRUM-24: MODEL COMPARISON REPORT ==========")
    print("IS4100 Group 4 — Customer Churn Prediction")
    print("=" * 55)

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_prob)
        dod       = "✅ PASS" if f1 >= F1_THRESHOLD else "❌ FAIL"

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "DoD (F1>0.75)": dod
        })

    df_report = pd.DataFrame(results)

    # Print formatted table
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'ROC-AUC':<12} {'DoD'}")
    print("-" * 70)
    for _, row in df_report.iterrows():
        print(f"{row['Model']:<25} {row['Accuracy']:<12} {row['F1 Score']:<12} {row['ROC-AUC']:<12} {row['DoD (F1>0.75)']}")

    return df_report


def recommend_best_model(df_report: pd.DataFrame) -> str:
    """
    SCRUM-24: Recommend the best model based on F1 score.
    """
    print(f"\n🏆 Model Recommendation:")
    print("-" * 55)

    # Filter models that passed DoD
    passed = df_report[df_report["DoD (F1>0.75)"] == "✅ PASS"]

    if passed.empty:
        print("❌ No model met the Definition of Done (F1 > 0.75)")
        print("   Action: Tune hyperparameters (SCRUM-21) and retry")
        return "None"

    # Pick highest F1 among passed models
    best = passed.loc[passed["F1 Score"].idxmax()]
    best_name = best["Model"]

    print(f"   ✅ Recommended: {best_name}")
    print(f"   F1 Score:  {best['F1 Score']}")
    print(f"   Accuracy:  {best['Accuracy']}")
    print(f"   ROC-AUC:   {best['ROC-AUC']}")

    # Business justification
    print(f"\n💼 Business Justification:")
    if "Random Forest" in best_name:
        print("   Random Forest is recommended because:")
        print("   • Handles non-linear churn patterns better")
        print("   • More robust to outliers in MonthlyCharges/tenure")
        print("   • Feature importance helps explain churn drivers")
        print("   • Higher F1 means fewer missed churners")
    else:
        print("   Logistic Regression is recommended because:")
        print("   • More interpretable for business stakeholders")
        print("   • Faster inference for real-time churn scoring")
        print("   • Coefficients directly show churn risk factors")

    return best_name


def save_report(df_report: pd.DataFrame,
                filepath: str = "reports/model_comparison_report.csv"):
    """
    SCRUM-24: Save comparison report to CSV for Appendix K.
    """
    df_report.to_csv(filepath, index=False)
    print(f"\n✅ Report saved to: {filepath}")
    print("   Use this for Appendix K — Closing Report & Lessons Learned")


def print_lessons_learned(best_model: str):
    """
    SCRUM-24: Print lessons learned for Appendix K of the report.
    """
    print(f"\n📋 Lessons Learned (Appendix K):")
    print("-" * 55)
    lessons = [
        "1. Data quality issues (TotalCharges as string) were caught early via SCRUM-15",
        "2. Class imbalance (~26% churn) required class_weight='balanced'",
        "3. StandardScaler was essential for Logistic Regression convergence",
        "4. Stratified train/test split preserved churn ratio correctly",
        "5. Random Forest outperformed Logistic Regression on F1",
        f"6. Final recommended model: {best_model}",
        "7. Hyperparameter tuning (SCRUM-21) improved F1 by tuning max_depth",
        "8. Agile/Scrum allowed us to pivot quickly when data issues arose",
    ]
    for lesson in lessons:
        print(f"   {lesson}")


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

    # Train both models
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train)
    }

    # Generate comparison report
    df_report = generate_comparison_report(models, X_test, y_test)

    # Recommend best model
    best = recommend_best_model(df_report)

    # Save report
    save_report(df_report)

    # Lessons learned
    print_lessons_learned(best)
