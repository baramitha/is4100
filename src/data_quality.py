# SCRUM-15: Data Quality Assessment
# Assignee: Naura
# Sprint: 1
# Epic: SCRUM-6 Data Engineering
 
import pandas as pd
import numpy as np
 
 
def assess_quality(df: pd.DataFrame) -> dict:
    """
    SCRUM-15: Assess the quality of the customer churn dataset.
    Checks for missing values, duplicates, outliers, and data type issues.
    """
    print("\n========== DATA QUALITY ASSESSMENT ==========")
    issues = []
 
    # 1. Missing values check
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print(f"\n❗ Missing Values Found:")
        print(missing_cols.to_string())
        issues.append(f"Missing values in: {missing_cols.index.tolist()}")
    else:
        print("\n✅ No missing values found")
 
    # 2. Duplicate rows check
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\n❗ Duplicate Rows: {duplicates}")
        issues.append(f"{duplicates} duplicate rows found")
    else:
        print("✅ No duplicate rows found")
 
    # 3. Outlier detection using IQR for numerical columns
    print(f"\n📊 Outlier Detection (IQR Method):")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_report = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_report[col] = len(outliers)
        if len(outliers) > 0:
            print(f"  ⚠️  {col}: {len(outliers)} outliers (range: {lower:.2f} to {upper:.2f})")
        else:
            print(f"  ✅ {col}: No outliers")
 
    # 4. Data type validation
    print(f"\n🔍 Data Type Validation:")
    # TotalCharges is often stored as object in churn datasets
    if "TotalCharges" in df.columns:
        non_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce").isnull().sum()
        if non_numeric > 0:
            print(f"  ⚠️  TotalCharges: {non_numeric} non-numeric values found — needs conversion")
            issues.append("TotalCharges contains non-numeric values")
        else:
            print("  ✅ TotalCharges: All numeric")
 
    # 5. Class imbalance check
    if "Churn" in df.columns:
        print(f"\n⚖️  Class Balance Check (Churn):")
        churn_pct = df["Churn"].value_counts(normalize=True) * 100
        print(churn_pct.round(2).to_string())
        if churn_pct.min() < 20:
            print("  ⚠️  Class imbalance detected — consider oversampling (SMOTE)")
            issues.append("Class imbalance in Churn column")
        else:
            print("  ✅ Class balance is acceptable")
 
    # Summary
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing.to_dict(),
        "duplicate_rows": int(duplicates),
        "outliers_per_column": outlier_report,
        "issues_found": issues,
        "quality_score": "PASS" if len(issues) == 0 else "NEEDS CLEANING"
    }
 
    print(f"\n🏁 Quality Score: {quality_report['quality_score']}")
    print(f"📋 Issues Found: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print("\n============================================\n")
 
    return quality_report
 
 
if __name__ == "__main__":
    from data_exploration import load_data
    df = load_data("data/raw/churn.csv")
    report = assess_quality(df)
