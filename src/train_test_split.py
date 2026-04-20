# SCRUM-18: Train/Test Split & Validation
# Assignee: Darryl
# Sprint: 1
# Epic: SCRUM-6 Data Engineering
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
 
 
def split_data(X, y, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    SCRUM-18: Split dataset into training and test sets.
    Default: 80% train / 20% test
    Uses stratified split to preserve Churn class ratio.
    """
    print("\n========== TRAIN/TEST SPLIT ==========")
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # preserve churn ratio in both sets
    )
 
    print(f"✅ Split complete (stratified):")
    print(f"   Training set:  {X_train.shape[0]} rows ({(1-test_size)*100:.0f}%)")
    print(f"   Test set:      {X_test.shape[0]} rows ({test_size*100:.0f}%)")
 
    # Validate class distribution is preserved
    train_churn_rate = y_train.mean() * 100
    test_churn_rate = y_test.mean() * 100
    print(f"\n⚖️  Churn Rate Validation:")
    print(f"   Training churn rate: {train_churn_rate:.2f}%")
    print(f"   Test churn rate:     {test_churn_rate:.2f}%")
 
    if abs(train_churn_rate - test_churn_rate) < 2:
        print("   ✅ Churn rates are balanced between train/test sets")
    else:
        print("   ⚠️  Churn rate difference is high — check stratification")
 
    print("\n======================================\n")
 
    return X_train, X_test, y_train, y_test
 
 
def validate_split(X_train, X_test, y_train, y_test):
    """
    SCRUM-18: Run validation checks on the split datasets.
    Ensures no data leakage and correct shapes.
    """
    print("\n🔍 Split Validation Checks:")
 
    # Shape checks
    assert X_train.shape[0] == y_train.shape[0], "❌ Train X/y size mismatch!"
    assert X_test.shape[0] == y_test.shape[0], "❌ Test X/y size mismatch!"
    assert X_train.shape[1] == X_test.shape[1], "❌ Feature count mismatch!"
    print("✅ Shape validation passed")
 
    # No missing values
    assert X_train.isnull().sum().sum() == 0, "❌ Missing values in X_train!"
    assert X_test.isnull().sum().sum() == 0, "❌ Missing values in X_test!"
    print("✅ No missing values in train/test sets")
 
    # No overlap (index check)
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    overlap = train_idx.intersection(test_idx)
    assert len(overlap) == 0, f"❌ Data leakage! {len(overlap)} overlapping rows"
    print("✅ No data leakage — train and test sets are fully separate")
 
    print("\n🏁 All validation checks passed!\n")
 
 
if __name__ == "__main__":
    from data_exploration import load_data
    from data_cleaning import clean_data
    from feature_engineering import engineer_features
 
    df = load_data("data/raw/churn.csv")
    df = clean_data(df)
    X, y, feature_names, scaler = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    validate_split(X_train, X_test, y_train, y_test)
