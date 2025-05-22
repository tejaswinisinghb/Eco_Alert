import pandas as pd
import joblib 
from scipy.stats import skew
df = pd.read_csv('flood.csv')
# Convert continuous target to binary based on a threshold
threshold = 0.5  # Example threshold; adjust as needed
df['FloodBinary'] = (df['FloodProbability'] > threshold).astype(int)

# Define the list of selected features
selected_features = [
    'TopographyDrainage',
    'Deforestation',
    'Urbanization',
    'IneffectiveDisasterPreparedness',
    'Encroachments',
    'DrainageSystems',
    'Landslides',
    'DeterioratingInfrastructure'
]

# Use the selected features for X
X = df[selected_features]

# Define your features and new target variable
X = df[selected_features[:-1]]  # Exclude original target variable
y = df['FloodBinary']  # Use the new binary target

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the new distribution of the target variable
print("Original target distribution:", y_train.value_counts())
print("Resampled target distribution:", y_train_resampled.value_counts())
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load your dataset into a DataFrame
data = pd.read_csv('flood.csv')  # Replace with the correct path to your dataset

# Features and target variable
X = data.drop('FloodProbability', axis=1)  # Assuming 'FloodProbability' is your target column
y = data['FloodProbability']

# Convert the target variable to binary
threshold = 0.5  # Set your threshold for classification
y_binary = (y > threshold).astype(int)  # Convert continuous to binary

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Apply SMOTE to the training data to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the parameter distribution for RandomizedSearchCV
param_distributions = {
    'max_depth': [3, 4, 5, 6, 7],          # Maximum depth of the trees
    'learning_rate': [0.01, 0.1, 0.2],     # Learning rate
    'n_estimators': [50, 100, 150],        # Number of boosting rounds
    'subsample': [0.5, 0.75, 1.0],         # Subsample ratio of the training instance
    'colsample_bytree': [0.5, 0.75, 1.0],  # Subsample ratio of columns when constructing each tree
    'gamma': [0, 0.1, 0.2, 0.3],           # Minimum loss reduction required to make a further partition
}

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', seed=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_model,
                                   param_distributions=param_distributions,
                                   n_iter=50,  # Number of parameter settings to sample
                                   scoring='accuracy',
                                   cv=3,        # 3-fold cross-validation
                                   verbose=2,
                                   n_jobs=-1,   # Use all available cores
                                   random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters and best score
best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# Train the model using the best parameters
best_xgb_model = random_search.best_estimator_

# Make predictions on the test set
y_pred_prob = best_xgb_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Evaluate the best model's performance
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

joblib.dump(best_xgb_model, 'best.joblib')
