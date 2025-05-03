import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np

data = pd.read_csv('your_data.csv')
X = data.drop('target_column', axis=1)
y = data['target_column']

# Handle categorical variables (if not using XGBoost's built-in support)
# For example, using pandas' get_dummies:
# X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)