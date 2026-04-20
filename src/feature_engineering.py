# SCRUM-17: Feature Engineering
# Assignee: Naura
# Sprint: 1
# Epic: SCRUM-6 Data Engineering
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
 
 
def engineer_features(df: pd.DataFrame, target_column: str = "Churn") -> tuple:
    """
    SCRUM-17: Engineer features for the customer churn model.
    - Separates features (X) and target (y)
    - Scales numerical features using StandardScaler
    Returns X (scaled features), y (target), feature names, scaler
    """
    print("\n========== FEATURE ENGINEERING ==========")
 
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
 
    X = df.drop(columns=[target_column])
    y = df[target_column]
 
    print(f"✅ Features (X): {X.shape[1]} columns")
    print(f"✅ Target (y): {target_column} — {y.value_counts().to_dict()}")
 
    # Identify numerical columns to scale
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n📊 Scaling numerical columns: {num_cols}")
 
    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
 
    print(f"✅ StandardScaler applied to {len(num_cols)} numerical columns")
    print(f"\n📐 Final feature matrix shape: {X_scaled.shape}")
 
    # Feature summary
    print(f"\n🔍 Feature Summary:")
    print(f"   Total features: {X_scaled.shape[1]}")
    print(f"   Numerical features: {len(num_cols)}")
    print(f"   Binary/Encoded features: {X_scaled.shape[1] - len(num_cols)}")
 
    print("\n==========================================\n")
 
    return X_scaled, y, X.columns.tolist(), scaler
 
 
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    SCRUM-17: Add new derived features to improve model performance.
    """
    print("\n🔧 Adding engineered features...")
 
    # Tenure groups — segment customers by loyalty
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 60, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
        )
        df["tenure_group"] = df["tenure_group"].astype(str)
        print("✅ Added tenure_group (loyalty segmentation)")
 
    # Monthly charge per tenure ratio
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        print("✅ Added charge_per_tenure ratio")
 
    # High value customer flag
    if "MonthlyCharges" in df.columns:
        threshold = df["MonthlyCharges"].quantile(0.75)
        df["is_high_value"] = (df["MonthlyCharges"] >= threshold).astype(int)
        print(f"✅ Added is_high_value flag (threshold: ${threshold:.2f}/month)")
 
    return df
 
 
if __name__ == "__main__":
    from data_exploration import load_data
    from data_cleaning import clean_data
 
    df = load_data("data/raw/churn.csv")
    df = clean_data(df)
    X, y, feature_names, scaler = engineer_features(df)
    print(f"Feature names: {feature_names[:5]}...")
