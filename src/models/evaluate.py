import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import os

# Get the absolute path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Define paths using BASE_DIR
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/cleaned_ckd_data.csv")
MODEL_PATH_REGRESSION = os.path.join(BASE_DIR, "models/ckd_stage_model_clinical.pkl")
MODEL_PATH_CLASSIFICATION = os.path.join(BASE_DIR, "models/affected_kidney_model_clinical.pkl")


def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    print("\nClassification Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")


def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRegression Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")


def evaluate_models():
    # Load the dataset
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Split features and targets
    X = df.drop(columns=["stage", "affected"])  # Features
    y_regression = df["stage"]  # Target for regression
    y_classification = df["affected"]  # Target for classification
    
    # Load models
    regression_model = joblib.load(MODEL_PATH_REGRESSION)
    classification_model = joblib.load(MODEL_PATH_CLASSIFICATION)
    
    # Evaluate models
    evaluate_regression_model(regression_model, X, y_regression)
    evaluate_classification_model(classification_model, X, y_classification)


if __name__ == "__main__":
    evaluate_models()