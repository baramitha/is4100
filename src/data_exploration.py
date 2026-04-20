# SCRUM-14: Data Exploration & Profiling
# Assignee: Darryl
# Sprint: 1
# Epic: SCRUM-6 Data Engineering
 
import pandas as pd
import numpy as np
 
def load_data(filepath: str) -> pd.DataFrame:
    """
    SCRUM-14: Load raw customer churn dataset from CSV.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
 
 
def explore_data(df: pd.DataFrame) -> dict:
    """
    SCRUM-14: Explore and profile the customer churn dataset.
    Returns a summary dictionary of key statistics.
    """
    print("\n========== DATA EXPLORATION REPORT ==========")
 
    # Basic shape
    print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
 
    # Column names and types
    print(f"\n📋 Columns & Data Types:")
    print(df.dtypes.to_string())
 
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(f"\n❗ Missing Values:")
    print(pd.DataFrame({"Count": missing, "Percentage (%)": missing_pct})
          .loc[missing > 0].to_string())
 
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔁 Duplicate Rows: {duplicates}")
 
    # Target variable distribution
    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts()
        churn_pct = df["Churn"].value_counts(normalize=True) * 100
        print(f"\n🎯 Target Variable (Churn) Distribution:")
        print(pd.DataFrame({"Count": churn_counts, "Percentage (%)": churn_pct.round(2)}).to_string())
 
    # Numerical summary
    print(f"\n📊 Numerical Columns Summary:")
    print(df.describe().round(2).to_string())
 
    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"\n🔤 Categorical Columns: {cat_cols}")
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values → {df[col].unique()[:5]}")
 
    summary = {
        "shape": df.shape,
        "missing_values": missing.to_dict(),
        "duplicates": int(duplicates),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "churn_distribution": df["Churn"].value_counts().to_dict() if "Churn" in df.columns else {}
    }
 
    print("\n============================================\n")
    return summary
 
 
if __name__ == "__main__":
    # Run exploration on the raw dataset
    df = load_data("data/raw/churn.csv")
    summary = explore_data(df)
