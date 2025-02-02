import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def add_interaction_features(df):
    """Creates interaction features from numerical columns."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = poly.fit_transform(df[numerical_cols])
    feature_names = poly.get_feature_names_out(numerical_cols)
    df_interactions = pd.DataFrame(interaction_features, columns=feature_names)
    df = pd.concat([df, df_interactions], axis=1)
    return df

def scale_features(df):
    """Scales numerical features using StandardScaler."""
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def feature_engineering(df):
    """Applies feature engineering steps."""
    df = add_interaction_features(df)
    df = scale_features(df)
    return df