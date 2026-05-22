# Car Price Prediction (Japan Imports)

This project build a machine learning system to predict used car prices.
The pipeline includes web scraping, data cleaning, feature engineering

The pipeline includes

- Web scraping
- Data cleaning
- Feature engineering
- Exploratory data analysis (EDA)
- Model training with cross-validation
- Hyperparameter tuning.

## Machine Learning Models

Models tested:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Gradient Boosting
- XGBoost

## Evaluation Metrics

Models are evaluated in:

### Log Space

- MAE
- R²

### USD Space (real-world interpretation)

- MAE (USD)
- R²

## How to Run (using uv)

```bash

git clone https://github.com/sirsona/japan-cars-import
cd japan-cars-import

uv sync
uv run jupyter notebook

```

