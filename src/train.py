# train.py
import logging
import os

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from config import DATA_PATH

# Setup simple logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_clean_data(DATA_PATH, target_col, remove_cols):
    """Loads data, handles null targets, clips outliers, and transforms target variable."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Drop target nulls
    df = df.dropna(subset=[target_col])

    # Clip extreme target outliers (1st and 99th percentiles)
    q_low = df[target_col].quantile(0.01)
    q_high = df[target_col].quantile(0.99)
    df = df[(df[target_col] >= q_low) & (df[target_col] <= q_high)]

    # Extract features and target
    y = np.log1p(df[target_col])
    X = df.drop(columns=remove_cols, errors="ignore")

    logging.info(f"Loaded data shape: {X.shape}. Target log-transformed successfully.")
    return X, y


def build_preprocessor(X):
    """Identifies column types and constructs the scikit-learn ColumnTransformer pipeline."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    logging.info(
        f"Features - Numerical: {len(num_cols)} | Categorical: {len(cat_cols)}"
    )

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )

    return ColumnTransformer(
        [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
    )


def evaluate_baseline_models(X_train, y_train, preprocessor):
    """Evaluates multiple models using multi-metric cross_validate in a single pass."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    summary = {}

    for name, model in models.items():
        full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        scores = cross_validate(
            full_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )

        summary[name] = {
            "MAE_mean": -scores["test_MAE"].mean(),
            "RMSE_mean": -scores["test_RMSE"].mean(),
            "R2_mean": scores["test_R2"].mean(),
        }

        logging.info(
            f"{name:<18} -> MAE: {summary[name]['MAE_mean']:.3f} | "
            f"RMSE: {summary[name]['RMSE_mean']:.3f} | R2: {summary[name]['R2_mean']:.3f}"
        )

    return pd.DataFrame(summary).T


def tune_hyperparameters(
    X_train,
    y_train,
    preprocessor,
    model_type,
    param_dist,
    n_iter=30,
):
    """Runs hyperparameter tuning for a designated estimator type inside a clean pipeline."""
    if model_type == "rf":
        base_model = RandomForestRegressor(random_state=42)
    elif model_type == "xgb":
        base_model = XGBRegressor(
            random_state=42, objective="reg:squarederror", n_jobs=-1
        )
    else:
        raise ValueError("Unsupported model_type. Choose 'rf' or 'xgb'.")

    pipe = Pipeline([("preprocessor", preprocessor), ("model", base_model)])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_on_test_set(model, X_test, y_test):
    """Generates predictions and scores the model on both Log Space and USD Space."""
    y_pred_log = model.predict(X_test)

    y_pred_usd = np.expm1(y_pred_log)
    y_test_usd = np.expm1(y_test)

    metrics = {
        "log_mae": float(mean_absolute_error(y_test, y_pred_log)),
        "log_r2": float(r2_score(y_test, y_pred_log)),
        "usd_mae": float(mean_absolute_error(y_test_usd, y_pred_usd)),
        "usd_r2": float(r2_score(y_test_usd, y_pred_usd)),
        "usd_rmse": float(root_mean_squared_error(y_test_usd, y_pred_usd)),
    }

    print("\n" + "=" * 30 + "\nEVALUATION RESULTS\n" + "=" * 30)
    print(f"LOG SPACE -> MAE: {metrics['log_mae']:.4f} | R2: {metrics['log_r2']:.4f}")
    print(
        f"USD SPACE -> MAE: ${metrics['usd_mae']:.2f} | "
        f"R2: {metrics['usd_r2']:.4f} | RMSE: ${metrics['usd_rmse']:.2f}\n"
    )

    return metrics


# CLI
if __name__ == "__main__":
    DATA_PATH = "../data/clean/5_beforward_clean_data.csv"
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
    MODEL_OUTPUT_DIR = "../models"

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    X, y = load_and_clean_data(DATA_PATH, TARGET_VAR, REMOVE_COLUMNS)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    preprocessor = build_preprocessor(X_train)

    logging.info("Starting baseline comparison...")
    baseline_results = evaluate_baseline_models(X_train, y_train, preprocessor)

    logging.info("Tuning XGBoost...")
    xgb_params = {
        "model__n_estimators": randint(100, 800),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    }

    xgb_search = tune_hyperparameters(
        X_train, y_train, preprocessor, "xgb", xgb_params, n_iter=20
    )

    evaluate_on_test_set(xgb_search.best_estimator_, X_test, y_test)

    dump(
        xgb_search.best_estimator_,
        os.path.join(MODEL_OUTPUT_DIR, "xgboost_cars_from_japan_model_v1.pkl"),
    )
