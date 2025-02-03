import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from src.data_preprocessing.clean_data import preprocess_data
import os

# Get the absolute path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Define paths using BASE_DIR
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/cleaned_ckd_data.csv")
MODEL_PATH_REGRESSION = os.path.join(BASE_DIR, "models/ckd_stage_model_clinical.pkl")
MODEL_PATH_CLASSIFICATION = os.path.join(BASE_DIR, "models/ckd_classification_model_clinical.pkl")


def train_models_clinical():
    """ Train both regression and classification models for CKD prediction. """

    # Load cleaned data
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Define Features and Targets
    X = df.drop(columns=["stage", "class"])  # All features except targets
    y_reg = df["stage"]  # Regression target
    y_clf = df["class"]  # Classification target

    # Split data
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dimensionality Reduction using PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # --- Train Regression Model (CKD Stage) ---
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_pca, y_train_reg)

    # --- Train Classification Model (Affected Kidneys) ---
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_pca, y_train_clf)

    # --- Evaluate Models ---
    y_pred_reg = reg_model.predict(X_test_pca)
    reg_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    y_pred_clf = clf_model.predict(X_test_pca)
    clf_acc = accuracy_score(y_test_clf, y_pred_clf)

    # Print Evaluation
    print(f"Regression Model RMSE (CKD Stage): {reg_rmse:.4f}")
    print(f"Classification Model Accuracy (CKD Presence/Not): {clf_acc:.4%}")

    # Save Models
    with open(MODEL_PATH_REGRESSION, "wb") as f:
        pickle.dump(reg_model, f)

    with open(MODEL_PATH_CLASSIFICATION, "wb") as f:
        pickle.dump(clf_model, f)

    print("Models saved successfully!")

if __name__ == "__main__":
    train_models_clinical()
