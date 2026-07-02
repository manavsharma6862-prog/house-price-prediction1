"""
preprocessing.py
Data preprocessing utilities for House Price Prediction
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


def load_dataset():
    """Load and return California Housing Dataset as a DataFrame."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, 'dataset', 'california_housing.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.copy()
    df.rename(columns={
        'MedInc': 'median_income',
        'HouseAge': 'house_age',
        'AveRooms': 'avg_rooms',
        'AveBedrms': 'avg_bedrooms',
        'Population': 'population',
        'AveOccup': 'avg_occupancy',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'MedHouseVal': 'median_house_value'
    }, inplace=True)
    return df


def handle_missing_values(df):
    """Check and handle missing values."""
    missing = df.isnull().sum()
    print("\n[INFO] Missing Values:\n", missing[missing > 0])
    df.dropna(inplace=True)
    return df


def remove_outliers(df, target_col='median_house_value'):
    """Remove outliers using IQR method on the target column."""
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    after = len(df)
    print(f"\n[INFO] Outliers removed: {before - after} rows")
    return df


def engineer_features(df):
    """Create new features from existing ones."""
    df = df.copy()
    df['rooms_per_person'] = df['avg_rooms'] / (df['avg_occupancy'] + 1e-9)
    df['bedrooms_ratio'] = df['avg_bedrooms'] / (df['avg_rooms'] + 1e-9)
    df['income_per_room'] = df['median_income'] / (df['avg_rooms'] + 1e-9)
    return df


def get_feature_target_split(df):
    """Return features X and target y."""
    feature_cols = [
        'median_income', 'house_age', 'avg_rooms', 'avg_bedrooms',
        'population', 'avg_occupancy', 'latitude', 'longitude',
        'rooms_per_person', 'bedrooms_ratio', 'income_per_room'
    ]
    X = df[feature_cols]
    y = df['median_house_value']
    return X, y


def scale_features(X_train, X_test):
    """Fit scaler on train, transform both train and test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


FEATURE_NAMES = [
    'median_income', 'house_age', 'avg_rooms', 'avg_bedrooms',
    'population', 'avg_occupancy', 'latitude', 'longitude',
    'rooms_per_person', 'bedrooms_ratio', 'income_per_room'
]

BASE_INPUT_FEATURES = [
    'median_income', 'house_age', 'avg_rooms', 'avg_bedrooms',
    'population', 'avg_occupancy', 'latitude', 'longitude'
]
