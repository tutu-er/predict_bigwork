import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np
from data import Data

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE


data = Data('STLF_DATA_IN_1.xls')
data_A = np.zeros((data.days, 10))
data_A[data.row_all_right] = pd.read_csv('A_ksvd_10.csv', header=None).to_numpy()
data_Weather = data.weather.to_numpy()

day_long = 2
day_long_last = 1

days_to_train = [i for i in range(365, data.days-1) if i in data.row_all_right and data.weekend_and_holiday_onehot[i] == 0]

X_data = []
X_date_data = []

for idx, day_index in enumerate(days_to_train):
    idx_type = data.weekend_and_holiday_onehot[day_index]

    type_list = np.array(data.weekend_and_holiday_onehot == idx_type)
    type_list[data.row_error] = False

    idx_same_type_day = []
    idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                 if value and index < day_index))
    for _ in range(day_long-1):
        idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                     if value and index < min(idx_same_type_day)))

    x_same_type_day = np.mean(data.np_day_pd_96[idx_same_type_day, :], axis=1).reshape(-1)

    idx_last_day = []
    idx_last_day.append(max(index for index in data.row_all_right
                                 if index < day_index))
    for _ in range(day_long_last-1):
        idx_last_day.append(max(index for index in data.row_all_right
                                 if index < min(idx_last_day)))

    x_last_day = np.mean(data.np_day_pd_96[idx_last_day, :], axis=1).reshape(-1)

    x_data_i = np.concatenate((x_same_type_day, x_last_day), axis=0).reshape(1, -1)
    X_data.append(x_data_i)

scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(np.concatenate(X_data, axis=0))
X_date = data.date_one_hot[days_to_train, :]
X_temperature = data.weather['平均温度'].to_numpy()[days_to_train].reshape(-1, 1)

X = np.concatenate((X, X_date, X_temperature), axis=1)

scaler_Y = MinMaxScaler()
y = scaler_Y.fit_transform(data.np_day_pd_96[days_to_train, :])

dtrain = xgb.DMatrix(X, label=y)
# 通用参数
params = {'objective': 'reg:squarederror', 'eval_metric': 'logloss', 'max_depth': 6, 'eta': 0.25, 'subsample': 0.9,
          'colsample_bytree': 0.9, 'seed': 42, 'nthread': 8}
model = xgb.XGBRegressor(**params)
model.fit(X, y, verbose=True)

y_pred = model.predict(X)

print(f'MSE:{MSE(scaler_Y.inverse_transform(y), scaler_Y.inverse_transform(y_pred))}')
pass

# Handle categorical variables (if not using XGBoost's built-in support)
# For example, using pandas' get_dummies:
# X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)