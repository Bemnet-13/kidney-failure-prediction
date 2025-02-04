import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import os

# Get the absolute path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Define paths using BASE_DIR
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/cleaned_ckd_data.csv")
MODEL_PATH_REGRESSION = os.path.join(BASE_DIR, "models/ckd_stage_model_user.pkl")
MODEL_PATH_CLASSIFICATION = os.path.join(BASE_DIR, "models/ckd_classification_model_user.pkl")

def train_models_user():
    """ Train both regression and classification models for CKD prediction. """

    # Load cleaned data
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Define Features and Targets
    selected_features = ["bp (Diastolic)", "bgr", "sod", "htn", "dm", "pe", "ane", "age"]
    X = df[selected_features]
    y_reg = df["stage"]  # Regression target
    y_clf = df["class"]  # Classification target
    
    # Encode categorical variables
    label_encoders = {}
    for col in ["htn", "dm", "pe", "ane", "class"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split data
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # Train Regression Model (CKD Stage)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train_reg)

    # Train Classification Model (CKD Status)
    clf_model = GaussianNB()
    clf_model.fit(X_train, y_train_clf)

    # Evaluate Models
    y_pred_reg = reg_model.predict(X_test)
    reg_mae = mean_absolute_error(y_test_reg, y_pred_reg)
    reg_mse = mean_squared_error(y_test_reg, y_pred_reg)

    y_pred_clf = clf_model.predict(X_test)
    clf_acc = accuracy_score(y_test_clf, y_pred_clf)

    # Print Evaluation
    print(f"Regression Model MAE (CKD Stage): {reg_mae:.4f}")
    print(f"Regression Model MSE (CKD Stage): {reg_mse:.4f}")
    print(f"Classification Model Accuracy (CKD Status): {clf_acc:.4%}")

    # Save Models
    with open(MODEL_PATH_REGRESSION, "wb") as f:
        pickle.dump(reg_model, f)

    with open(MODEL_PATH_CLASSIFICATION, "wb") as f:
        pickle.dump(clf_model, f)

    print("Models saved successfully!")

if __name__ == "__main__":
    train_models_user()