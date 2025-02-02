import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """Handles missing values by filling or dropping where necessary."""
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

def convert_data_types(df):
    """Converts data types where necessary."""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

def encode_categorical_features(df):
    """Encodes categorical features using LabelEncoder."""
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def preprocess_data(filepath):
    """Pipeline to load and clean the dataset."""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df, encoders = encode_categorical_features(df)
    return df, encoders
