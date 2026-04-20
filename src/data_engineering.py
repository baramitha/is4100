Content:
# SCRUM-14: Data Exploration & Profiling
# SCRUM-15: Data Quality Assessment
# SCRUM-16: Data Cleaning Script

import pandas as pd

def load_data(filepath):
    """Load raw dataset"""
    df = pd.read_csv(filepath)
    return df

def profile_data(df):
    """SCRUM-14: Explore and profile dataset"""
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print("Data types:\n", df.dtypes)
    return df.describe()

def assess_quality(df):
    """SCRUM-15: Data quality assessment"""
    quality_report = {
        "total_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum()
    }
    return quality_report

def clean_data(df):
    """SCRUM-16: Data cleaning"""
    df = df.drop_duplicates()
    df = df.dropna()
    return df
