# SCRUM-16: Data Cleaning Script
# Assignee: Naura
# Sprint: 1
# Epic: SCRUM-6 Data Engineering
 
import pandas as pd
import numpy as np
 
 
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    SCRUM-16: Clean the customer churn dataset.
    Handles missing values, duplicates, type conversion, and encoding.
    """
    print("\n========== DATA CLEANING ==========")
    original_shape = df.shape
 
    # 1. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"✅ Removed duplicates: {before - after} rows dropped")
 
    # 2. Drop customerID — not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("✅ Dropped customerID column (not a feature)")
 
    # 3. Convert TotalCharges to numeric
    # In Telco Churn dataset, TotalCharges has spaces for new customers
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill NaN TotalCharges with 0 (new customers with no charges yet)
        filled = df["TotalCharges"].isnull().sum()
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
        print(f"✅ Converted TotalCharges to numeric ({filled} nulls filled with 0)")
 
    # 4. Handle remaining missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    for col in missing_cols.index:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"✅ Filled missing '{col}' with mode: {df[col].mode()[0]}")
        else:
            df[col] = df[col].fillna(df[col].median())
            print(f"✅ Filled missing '{col}' with median: {df[col].median():.2f}")
 
    # 5. Encode binary Yes/No columns
    binary_cols = [col for col in df.columns
                   if df[col].dtype == "object"
                   and set(df[col].dropna().unique()).issubset({"Yes", "No"})]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"✅ Binary encoded columns: {binary_cols}")
 
    # 6. Encode Churn target column
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        print("✅ Encoded Churn: Yes=1, No=0")
 
    # 7. Encode SeniorCitizen — already 0/1 in Telco dataset, skip if so
    # 8. One-hot encode remaining categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"✅ One-hot encoded: {cat_cols}")
 
    print(f"\n📐 Shape before cleaning: {original_shape}")
    print(f"📐 Shape after cleaning:  {df.shape}")
    print("\n====================================\n")
 
    return df
 
 
def save_clean_data(df: pd.DataFrame, filepath: str = "data/processed/churn_clean.csv"):
    """Save the cleaned dataset to processed folder."""
    df.to_csv(filepath, index=False)
    print(f"✅ Clean data saved to: {filepath}")
 
 
if __name__ == "__main__":
    from data_exploration import load_data
    df = load_data("data/raw/churn.csv")
    df_clean = clean_data(df)
    save_clean_data(df_clean)
