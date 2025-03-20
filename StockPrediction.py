# StockPrediction.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from Pipeline import fetch_stock_data, add_features
import numpy as np

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

if __name__ == "__main__":
    # Fetch and prepare data
    raw_data = fetch_stock_data()
    df = add_features(raw_data)

    # Features and target
    features = ['MA5', 'MA10', 'MA20', 'Volatility', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5']
    X = df[features]
    y = df['Close']

    # Time-series train-test split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Random Forest Model
    rf_model = train_random_forest(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # XGBoost Model
    xgb_model = train_xgboost(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    # Evaluation
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mape = mean_absolute_percentage_error(y_test, rf_preds) * 100

    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds) * 100

    print(f"Random Forest - RMSE: {rf_rmse:.2f}, MAPE: {rf_mape:.2f}%")
    print(f"XGBoost - RMSE: {xgb_rmse:.2f}, MAPE: {xgb_mape:.2f}%")
