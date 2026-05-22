#
# Central configuration for data paths and model settings.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_cars_from_japan_model_v1.pkl"

DATA_PATH = BASE_DIR / "data" / "clean" / "5_beforward_clean_data.csv"
