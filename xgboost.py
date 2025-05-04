import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np
from data import Data


data = Data('STLF_DATA_IN_1.xls')
data_A = np.zeros((data.days, 10))
data_A[data.row_all_right] = pd.read_csv('A_ksvd_10.csv', header=None).to_numpy()
data_Weather = data.weather.to_numpy()

day_long = 2
day_long_diff = 2

days_to_predict = [i for i in range(365, data.days-1) if i in data.row_all_right and data.weekend_and_holiday_onehot[i] == 0]

X_data = []

for idx, day_index in enumerate(days_to_predict):
    idx_type = data.weekend_and_holiday_onehot[day_index]

    type_list = np.array(data.weekend_and_holiday_onehot == idx_type)
    type_list[data.row_error] = False

    idx_same_type_day = []
    idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                 if value and index < day_index))
    for _ in range(day_long-1):
        idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                     if value and index < min(idx_same_type_day)))

    x_same_type_day = data_A[idx_same_type_day, :].reshape(-1)

    idx_diff_type_day = []
    idx_diff_type_day.append(max(index for index, value in enumerate(type_list)
                                 if index < day_index and not value))
    for _ in range(day_long-1):
        idx_diff_type_day.append(max(index for index, value in enumerate(type_list)
                                     if index < min(idx_diff_type_day) and not value))

    x_diff_type_day = data_A[idx_diff_type_day, :].reshape(-1)

    x_data_i = np.concatenate((x_same_type_day, x_diff_type_day), axis=0).reshape(1, -1)
    X_data.append(x_data_i)

X = np.concatenate(X_data, axis=0)

pass

# Handle categorical variables (if not using XGBoost's built-in support)
# For example, using pandas' get_dummies:
# X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)