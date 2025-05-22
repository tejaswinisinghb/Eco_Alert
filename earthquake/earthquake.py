import pandas as pd
import joblib 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np

# Read CSV file
df = pd.read_csv('earthquake.csv', delimiter=',')

# Select relevant columns
X = df[['latitude', 'longitude', 'depth', 'sig', 'mmi', 'cdi']]
y = df['magnitude']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize a random forest regressor with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the regressor to the training data
rf.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred_rf = rf.predict(X_test)

# Evaluate the performance of the model using mean squared error and R^2 score
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('Random Forest - Mean Squared Error:', mse_rf)
print('Random Forest - R^2 Score:', r2_rf)

# Define an acceptable margin
margin = 0.28

# Calculate the percentage of predictions within the acceptable margin
accurate_predictions_rf = np.abs(y_pred_rf - y_test) <= margin
accuracy_rf = np.mean(accurate_predictions_rf) * 100
print(f'Random Forest - Accuracy within ±{margin} magnitude: {accuracy_rf:.2f}%')

# Initialize the XGBoost Regressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Fit the model to the training data
xg_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xg_reg.predict(X_test)

# Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print('XGBoost - Mean Squared Error:', mse_xgb)
print('XGBoost - R^2 Score:', r2_xgb)
print('XGBoost - Mean Absolute Error:', mae_xgb)

# Calculate accuracy within a specified margin for XGBoost
accurate_predictions_xgb = np.abs(y_pred_xgb - y_test) <= margin
accuracy_xgb = np.mean(accurate_predictions_xgb) * 100
print(f'XGBoost - Accuracy within ±{margin} magnitude: {accuracy_xgb:.2f}%')

# Define the model for hyperparameter tuning
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xg_reg,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Train model with the best parameters from grid search
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred_best = best_model.predict(X_test)

# Calculate accuracy within a specified margin for the best model
accurate_predictions_best = np.abs(y_pred_best - y_test) <= margin
accuracy_best = np.mean(accurate_predictions_best) * 100

# Additional metrics for the best model
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f'Best Model - Accuracy within ±{margin} magnitude: {accuracy_best:.2f}%')
print('Best Model - Mean Absolute Error:', mae_best)
print('Best Model - R^2 Score:', r2_best)

# Create a DataFrame for new data
new_data = pd.DataFrame({
    'latitude': [34.05],
    'longitude': [-118.25],
    'depth': [10.0],
    'sig': [500],
    'mmi': [30],
    'cdi': [20]
})

# Predict using the trained best model
predicted_values = best_model.predict(new_data)

# Output the predictions
print('\nPredicted Magnitude:', predicted_values)

joblib.dump(best_model, 'random.joblib')
