import os
from src.data_preprocessing.clean_data import preprocess_data
from src.data_preprocessing.feature_engineering import feature_engineering
from src.models.train_clinical import train_models_clinical
from src.models.evaluate import evaluate_models
from src.models.train_user import train_models_user


# Define paths
RAW_DATA_PATH = os.path.join("data", "raw", "ckd-dataset-v2.csv")
CLEANED_DATA_PATH = os.path.join("data", "processed", "cleaned_ckd_data.csv")
ENGINEERED_DATA_PATH = os.path.join("data", "processed", "engineered_ckd_data.csv")

def main():
    print("🚀 Starting Data Preprocessing...")

    # Step 1: Data Cleaning
    cleaned_df, encoders = preprocess_data(RAW_DATA_PATH)
    print("✅ Data Cleaning Complete.")
    
    # Save the cleaned data
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"📁 Cleaned data saved to {CLEANED_DATA_PATH}")

    # Step 2: Feature Engineering
    engineered_df = feature_engineering(cleaned_df)
    print("✅ Feature Engineering Complete.")

    # Step 3: Save Processed Data
    engineered_df.to_csv(ENGINEERED_DATA_PATH, index=False)
    print(f"📁 Engineered data saved to {ENGINEERED_DATA_PATH}")


    # Step 4: Model Training(clinical)
    print("Starting training process...")
    train_models_clinical()
    print("Training completed successfully!")

    # Step 5: Model Training(user)
    print("Starting training process...")
    train_models_user()
    print("Training completed successfully!")

    
    # Step 5: Model Evaluation
    # print("--------------------")
    # print("Starting model evalutation")
    # evaluate_models()
    # print("Model Evaluation Completed successfully")

if __name__ == "__main__":
    main()
