import numpy as np
import pandas as pd

from data import Data

from sklearn.metrics import mean_squared_error as MSE

class MRSM:
    def __init__(self, x, date, row_error, pu_type, day_type):
        self.x = x
        self.date = date
        self.row_error = row_error
        self.day_type = day_type

        self.x_base, self.x_pu = Data.deal_pu(data=x, row_error=row_error, type=pu_type)
        pass

    def predict_pu(self, target_day, alpha=0.1, N=14):

        predict_pu = np.zeros((1, 96))

        if isinstance(target_day, str):
            idx_loc = self.date.eq(pd.to_datetime(target_day, format='%Y%m%d'))
            idx = np.where(idx_loc)[0][0]
        elif isinstance(target_day, int):
            idx = target_day
        else:
            idx = round(target_day)

        idx_lower_bound = int(max(0, idx - N))

        id_now = 0
        sum_now = 0

        idx_type = self.day_type[idx]

        type_list = np.array(self.day_type == idx_type)
        type_list[self.row_error] = False

        time_interval = 6
        idx_same_type_day = []
        idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                     if value and index < idx-time_interval))
        for i in range(int(N/time_interval)):
            idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                         if value and index < min(idx_same_type_day) - time_interval))

        for i in range(len(idx_same_type_day)-1, - 1, -1):
            if idx_same_type_day[i] not in self.row_error:
                predict_pu += alpha * ((1 - alpha) ** id_now) * self.x_pu[idx_same_type_day[i], :]
                sum_now += alpha * ((1 - alpha) ** id_now)
                id_now += 1

        for i in range(idx, idx_lower_bound - 1, -1):
            if i not in self.row_error and i not in idx_same_type_day:
                predict_pu += alpha * ((1 - alpha) ** id_now) * self.x_pu[i, :]
                sum_now += alpha * ((1 - alpha) ** id_now)
                id_now += 1

        predict_pu = predict_pu / sum_now

        return predict_pu

    def predict_base(self, target_day, alpha=0.1, N=7):

        if isinstance(target_day, str):
            idx_loc = self.date.eq(pd.to_datetime(target_day, format='%Y%m%d'))
            idx = np.where(idx_loc)[0][0]
        elif isinstance(target_day, int):
            idx = target_day
        else:
            idx = round(target_day)

        idx_upper_bound_1 = idx-1
        idx_lower_bound_1 = int(max(0, idx-1 - N))

        idx_type = self.day_type[idx]

        type_list = np.array(self.day_type == idx_type)
        type_list[self.row_error] = False

        time_interval = 6
        idx_same_type_day = []
        idx_same_type_day.append(max(index for index, value in enumerate(type_list)
                                     if value and index < idx - time_interval))

        idx_upper_bound_2 = int(max(0, idx_same_type_day[0] - 1))
        idx_lower_bound_2 = int(max(0, idx_same_type_day[0] - 1 - N))
        x_base_former = self.x_base[idx_same_type_day[0]]

        id_now = 0
        sum_now = 0
        A_1 = 0
        for i in range(idx_upper_bound_1, idx_lower_bound_1 - 1, -1):
            if i not in self.row_error:
                A_1 += alpha * ((1 - alpha) ** id_now) * self.x_base[i]
                sum_now += alpha * ((1 - alpha) ** id_now)
                id_now += 1

        A_1 = A_1 / sum_now

        id_now = 0
        sum_now = 0
        A_2 = 0
        for i in range(idx_upper_bound_2, idx_lower_bound_2 - 1, -1):
            if i not in self.row_error:
                A_2 += alpha * ((1 - alpha) ** id_now) * self.x_base[i]
                sum_now += alpha * ((1 - alpha) ** id_now)
                id_now += 1

        A_2 = A_2 / sum_now

        predict_base = A_1/A_2*x_base_former

        return predict_base



if __name__ == '__main__':
    data = Data('STLF_DATA_IN_1.xls')

    """ MRSM """
    mrsm = MRSM(x=data.np_day_pd_96, date=data.date_info, row_error=data.row_error,
                pu_type='mean', day_type=data.weekend_and_holiday_onehot)

    data_pred_pu = np.zeros((data.days, data.T))
    data_pred_base = np.zeros(data.days)
    data_pred = np.zeros((data.days, data.T))

    days_to_predict = [i for i in range(30, data.days) if i in data.row_all_right]
    for i in days_to_predict:
        data_pred_pu[i, :] = mrsm.predict_pu(target_day=i, alpha=0.25, N=14)
        data_pred_base[i] = mrsm.predict_base(target_day=i, alpha=0.4, N=7)
        data_pred[i, :] = data_pred_base[i] * data_pred_pu[i, :]

    print(data.np_date[-2])
    print(MSE(mrsm.x_pu[-2, :], data_pred_pu[-2, :]))

    print(MSE(mrsm.x_pu[days_to_predict, :], data_pred_pu[days_to_predict, :]))

    print(data.np_date[-2])

    print(data_pred_base[-2], mrsm.x_base[-2])

    print(MSE(mrsm.x[-2, :], data_pred[-2, :]))
    print(MSE(mrsm.x[-2, :], data_pred[-2, :])/
          MSE(mrsm.x[-2, :], np.zeros(data.T)))

    print(MSE(mrsm.x[days_to_predict, :], data_pred[days_to_predict, :]))
    print(MSE(mrsm.x[days_to_predict, :], data_pred[days_to_predict, :])/
          MSE(mrsm.x[days_to_predict, :], np.zeros((len(days_to_predict), data.T))))



    pass






