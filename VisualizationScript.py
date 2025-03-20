# VisualizationScript.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from StockPrediction import fetch_stock_data, add_features, train_random_forest, train_xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Fetch and prepare the data
raw_data = fetch_stock_data()
df = add_features(raw_data)

features = ['MA5', 'MA10', 'MA20', 'Volatility', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5']
X = df[features]
y = df['Close']

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train models
rf_model = train_random_forest(X_train, y_train)
rf_preds = rf_model.predict(X_test)

xgb_model = train_xgboost(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Prepare results for plotting
results = pd.DataFrame({
    'Actual': y_test.values.ravel(),
    'RandomForest': rf_preds,
    'XGBoost': xgb_preds
}, index=y_test.index)


# 3D Plot - Actual vs Predictions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Convert DateTime index to numeric for 3D plot
time_numeric = np.arange(len(results.index))

# 3D Plot - Actual vs Predictions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(time_numeric, results['Actual'], results['RandomForest'], color='blue', label='Random Forest')
ax.plot(time_numeric, results['Actual'], results['XGBoost'], color='red', label='XGBoost')

ax.set_xlabel('Time (Index)')
ax.set_ylabel('Actual Price')
ax.set_zlabel('Predicted Price')
ax.set_title('3D Comparison: Actual vs Predicted')
ax.legend()

plt.tight_layout()
plt.show()

#  Save the 3D plot in results folder
fig.savefig('results/3D_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')


# Model Performance Metrics
rf_rmse = np.sqrt(mean_squared_error(results['Actual'], results['RandomForest']))
xgb_rmse = np.sqrt(mean_squared_error(results['Actual'], results['XGBoost']))
rf_mape = mean_absolute_percentage_error(results['Actual'], results['RandomForest']) * 100
xgb_mape = mean_absolute_percentage_error(results['Actual'], results['XGBoost']) * 100

print(f"Random Forest - RMSE: {rf_rmse:.2f}, MAPE: {rf_mape:.2f}%")
print(f"XGBoost - RMSE: {xgb_rmse:.2f}, MAPE: {xgb_mape:.2f}%")



# 2D Plot - Actual vs Predictions
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Actual'], label='Actual Price', color='black', linewidth=2)
plt.plot(results.index, results['RandomForest'], label='Random Forest', color='blue', linestyle='--')
plt.plot(results.index, results['XGBoost'], label='XGBoost', color='red', linestyle='--')

plt.title('2D Comparison: Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/2D_Actual_vs_Predicted.png', dpi=300, bbox_inches='tight')
plt.show()
