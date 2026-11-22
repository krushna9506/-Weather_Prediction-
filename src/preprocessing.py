"""
Module: preprocessing.py
Stage D: Data Preparation and Preprocessing [2 Marks - CO1, L3]
Aligned with WEATHER-5K Global Station Architecture (60,000 observations)
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any, List

def clean_and_preprocess_data(
    raw_df: pd.DataFrame,
    save_dir: str = "data/processed",
    artifacts_dir: str = "models"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, StandardScaler, LabelEncoder, List[str], Dict[str, str]]:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    df = raw_df.copy()
    
    # 1. Missing Value Audit & Imputation
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Outlier Treatment via IQR Clipping on Sensor Telemetry
    num_sensor_cols = [
        'Temperature', 'Dewpoint', 'Humidity', 'Wind Speed', 'Wind Direction',
        'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)',
        'Latitude', 'Longitude', 'Altitude (m)'
    ]
    for col in num_sensor_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # 3. Target Label Encoding
    le = LabelEncoder()
    df['Weather_Type_Encoded'] = le.fit_transform(df['Weather Type'])
    
    # Save readable cleaned dataset
    readable_path = os.path.join(save_dir, "weather_cleaned_readable.csv")
    df.to_csv(readable_path, index=False)
    df.to_csv("weather_cleaned_readable.csv", index=False)

    # 4. Categorical Feature Encoding (One-Hot Encoding with drop_first=True)
    nominal_cols = [c for c in ['Cloud Cover', 'Season', 'Location', 'Region'] if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # Columns to exclude from predictor matrix (identifiers, forecast horizons, target)
    metadata_cols = [
        'Station_ID', 'Station_Name', 'Forecast_24h_TMP', 'Forecast_72h_TMP', 
        'Forecast_24h_SLP', 'Weather Type', 'Weather_Type_Encoded'
    ]
    feature_cols = [c for c in df_encoded.columns if c not in metadata_cols]
    
    X = df_encoded[feature_cols]
    y = df_encoded['Weather_Type_Encoded']

    # 5. Feature Standardization via StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    # Preprocessed dataframe for storage
    preprocessed_df = pd.concat([X, df_encoded['Weather_Type_Encoded']], axis=1)
    prep_path = os.path.join(save_dir, "weather_cleaned_preprocessed.csv")
    preprocessed_df.to_csv(prep_path, index=False)
    preprocessed_df.to_csv("weather_cleaned_preprocessed.csv", index=False)

    # Save artifacts
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, os.path.join(artifacts_dir, "label_encoder.pkl"))
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(feature_cols, os.path.join(artifacts_dir, "feature_columns.pkl"))
    joblib.dump(feature_cols, "feature_columns.pkl")

    justifications = {
        "WEATHER-5K Feature Integration": "Incorporates core telemetry from WEATHER-5K: Temperature (TMP), Dewpoint (DPT), Wind Direction (WDIR 0-360°), Wind Speed (WSPD), and Pressure (SLP) alongside geodetic coordinates (Latitude, Longitude, Altitude).",
        "IQR Outlier Clipping": "Suppresses extreme sensor hardware glitches while preserving valid severe storm observations and high-altitude barometric depressions.",
        "One-Hot Encoding (drop_first=True)": "Converts nominal categories (Cloud Cover, Season, Location, Region) into orthogonal binary indicators, eliminating the dummy variable trap.",
        "StandardScaler (Z-Score Normalization)": "Normalizes continuous sensor readings to zero mean (μ=0) and unit variance (σ=1) (z = (x - μ) / σ), preventing Elevation (0-3500m) and Wind Direction (0-360°) from numerically dominating Temperature (-35 to +50°C)."
    }

    return preprocessed_df, X_scaled, y, scaler, le, feature_cols, justifications

if __name__ == "__main__":
    from data_collection import load_dataset
    df, _ = load_dataset()
    prep_df, X_sc, y, scaler, le, cols, justs = clean_and_preprocess_data(df)
    print(f"Preprocessed Dataset Shape: {prep_df.shape}")
    print(f"Model Feature Columns ({len(cols)}): {cols}")
