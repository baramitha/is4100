Content:
# SCRUM-17: Feature Engineering

import pandas as pd
from sklearn.preprocessing import StandardScaler

def engineer_features(df, target_column):
    """SCRUM-17: Create features for model"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
