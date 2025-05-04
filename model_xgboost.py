import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np
from data import Data
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE

class XGB:
    def __init__(self, data, params):
        self.model = xgb.XGBRegressor(**params)
        self.data = data

    def fit(self, X, Y):
        self.model.fit(X, Y, verbose=True)

    def predict(self, X):
        return self.model.predict(X)

    def construct_data_by_index(self, day_long, day_long_last, indexs, scaler_X=None, scaler_Y=None):
        X_data = []
        for day_index in indexs:
            """ same type day (mean) """
            idx_type = self.data.weekend_and_holiday_onehot[day_index]
            type_list = np.array(self.data.weekend_and_holiday_onehot == idx_type)
            type_list[self.data.row_error] = False
            idx_same_type_day = []
            idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                         if value and index < day_index))
            for _ in range(day_long - 1):
                idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                             if value and index < min(idx_same_type_day)))
            x_same_type_day_mean = np.mean(self.data.np_day_pd_96[idx_same_type_day, :], axis=1)
            x_same_type_day_max = np.max(self.data.np_day_pd_96[idx_same_type_day, :], axis=1)
            x_same_type_day_min = np.min(self.data.np_day_pd_96[idx_same_type_day, :], axis=1)
            x_same_type_day = np.concatenate((x_same_type_day_mean, x_same_type_day_max, x_same_type_day_min), axis=0).reshape(-1)

            """ former day (mean) """
            idx_last_day = []
            if day_long_last > 0:
                idx_last_day.append(max(index for index in self.data.row_all_right
                                        if index < day_index))
            for _ in range(day_long_last - 1):
                idx_last_day.append(max(index for index in self.data.row_all_right
                                        if index < min(idx_last_day)))
            x_last_day_mean = np.mean(self.data.np_day_pd_96[idx_last_day, :], axis=1)
            x_last_day_max = np.max(self.data.np_day_pd_96[idx_last_day, :], axis=1)
            x_last_day_min = np.min(self.data.np_day_pd_96[idx_last_day, :], axis=1)
            x_last_day = np.concatenate((x_last_day_mean, x_last_day_max, x_last_day_min),
                                             axis=0).reshape(-1)

            x_data_i = np.concatenate((x_same_type_day, x_last_day), axis=0).reshape(1, -1)
            X_data.append(x_data_i)
        if scaler_X is None:
            scaler_X = MinMaxScaler()
            X = scaler_X.fit_transform(np.concatenate(X_data, axis=0))
        else:
            X = scaler_X.transform(np.concatenate(X_data, axis=0))
        X_date = np.concatenate((self.data.date_one_hot[indexs, :],
                                 self.data.weekend_and_holiday_onehot[indexs].reshape(-1, 1),
                                 self.data.spring_fes_onehot[indexs].reshape(-1, 1)), axis= 1)
        X_temperature = self.data.weather['平均温度'].to_numpy()[indexs].reshape(-1, 1)
        X = np.concatenate((X, X_date, X_temperature), axis=1)

        if scaler_Y is None:
            scaler_Y = MinMaxScaler()
            Y = scaler_Y.fit_transform(self.data.np_day_pd_96[indexs, :])
        else:
            Y = scaler_Y.transform(self.data.np_day_pd_96[indexs, :])

        return scaler_X, X, scaler_Y, Y

if __name__ == "__main__":
    data = Data('STLF_DATA_IN_1.xls')
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_depth': 8, 'eta': 0.10, 'subsample': 0.9,
              'colsample_bytree': 0.9, 'seed': 42, 'nthread': 8}
    m = XGB(data, params)

    day_long = 3
    day_long_last = 0

    days_to_train = [i for i in range(365, data.days - 7) if
                     i in data.row_all_right and data.weekend_and_holiday_onehot[i] == 0]
    # days_to_train = [i for i in range(365, data.days - 7) if
    #                  i in data.row_all_right ]
    scalerX, X, scalerY, Y = m.construct_data_by_index(day_long=day_long, day_long_last=day_long_last, indexs=days_to_train)

    m.fit(X, Y)

    Y_pred = m.predict(X)
    print(f'MSE:{MSE(scalerY.inverse_transform(Y), scalerY.inverse_transform(Y_pred))}')

    days_to_test = [data.days - 2]
    _, X_test, _, Y_test = m.construct_data_by_index(day_long=day_long, day_long_last=day_long_last,
                                                       indexs=days_to_test, scaler_X=scalerX, scaler_Y=scalerY)
    Y_test_pred = m.predict(X_test)
    print(f'MSE:{MSE(scalerY.inverse_transform(Y_test), scalerY.inverse_transform(Y_test_pred))}')

    print(f'MSE:{MSE(data.np_day_pd_96[data.days-2, :], data.np_day_pd_96[data.days-5, :]*1.05)}')

    plt.plot(np.linspace(1, 96, 96), scalerY.inverse_transform(Y_test).T)
    plt.plot(np.linspace(1, 96, 96), scalerY.inverse_transform(Y_test_pred).T)
    plt.show()

    pass


