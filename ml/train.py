import argparse
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run(data_path: Path, model_output_path: Path):
    logging.info("Starting training process...")

    logging.info(f"Reading data from {data_path}")
    df = pd.read_csv(data_path)

    logging.info("Shuffling and splitting data...")
    df_shuffle = df.sample(frac=1, random_state=123)

    X = df_shuffle.drop(
        ["outcome_profit", "outcome_damage_inc", "outcome_damage_amount"], axis=1
    )
    y = df_shuffle["outcome_damage_inc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    logging.info("Standardizing numerical features...")
    num_feat = X_train.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    scaler.fit(X_train[num_feat])

    X_train_stan = X_train.copy()
    X_test_stan = X_test.copy()
    X_train_stan[num_feat] = scaler.transform(X_train[num_feat])
    X_test_stan[num_feat] = scaler.transform(X_test[num_feat])

    logging.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=123)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_stan, y_train)

    logging.info("Training SVC model...")
    svc = SVC(random_state=123, probability=True)
    svc.fit(X_train_smote, y_train_smote)

    logging.info(f"Saving scaler and model to {model_output_path}...")
    os.makedirs(model_output_path, exist_ok=True)
    joblib.dump(scaler, model_output_path / "scaler.joblib")
    joblib.dump(svc, model_output_path / "svc_model.joblib")

    logging.info("Training process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the training CSV file."
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        required=True,
        help="Path to save the trained model and scaler.",
    )
    args = parser.parse_args()

    run(Path(args.data_path), Path(args.model_output_path))
