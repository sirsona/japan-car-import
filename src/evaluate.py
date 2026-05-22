# evaluate.py

import os

from joblib import load
from sklearn.model_selection import train_test_split

from config import DATA_PATH, MODEL_PATH
from train import (
    evaluate_on_test_set,
    load_and_clean_data,
)

# =========================
# Configuration
# =========================

# DATA_PATH = "../data/clean/5_beforward_clean_data.csv"
#
# MODEL_PATH = "../models/xgboost_cars_from_japan_model_v1.pkl"

TARGET_VAR = "price_usd"

REMOVE_COLUMNS = [
    "price_usd",
    "total_price_usd",
    "shipping_cost",
    "vehicle_id",
    "ref_no",
    "vehicle_url",
    "engine_code",
]


def load_test_split():
    """
    Recreate the same test split used during training.
    """

    X, y = load_and_clean_data(DATA_PATH, TARGET_VAR, REMOVE_COLUMNS)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_test, y_test


def main():
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load trained model pipeline
    model_pipeline = load(MODEL_PATH)

    # Load test data
    X_test, y_test = load_test_split()

    # Evaluate model
    evaluate_on_test_set(model_pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
