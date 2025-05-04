import numpy as np
import pandas as pd
import datetime
import holidays
import chinese_calendar
import lunardate

from matplotlib import pyplot as plt

class Data:
    def __init__(self, data_name):
        self.df_pd = pd.read_excel(data_name, sheet_name='负荷数据', header=None)
        self.df_weather = pd.read_excel(data_name, sheet_name='气象数据', header=None)

        (self.df_date, self.df_day_pd_96,
         self.row_nan_indices, self.row_zero_indices, self.row_error, self.row_all_right) = self.deal_pd_data()

        self.np_date = np.array(self.df_date)
        self.np_day_pd_96 = np.array(self.df_day_pd_96)

        self.days = np.size(self.np_day_pd_96, axis=0)
        self.T = np.size(self.np_day_pd_96, axis=1)

        self.weather = self.deal_weather_data()

        (self.weekend, self.weekend_onehot, self.weekend_and_holiday, self.weekend_and_holiday_onehot,
         self.spring_fes, self.spring_fes_onehot,self.date_one_hot, self.date_info) = self.deal_time_onehot()

    def deal_pd_data(self):
        df_date = self.df_pd.iloc[:, 0]
        df_day_96 = self.df_pd.iloc[:, 1:]

        rows_with_nan = df_day_96.isna().any(axis=1)
        row_nan_indices = np.where(rows_with_nan)[0]

        rows_with_zero = df_day_96.eq(0).any(axis=1)
        row_zero_indices = np.where(rows_with_zero)[0]

        row_error = np.concatenate((row_nan_indices, row_zero_indices), axis=0)

        row_all = np.array([i for i in range(len(df_date))])
        row_all_right = np.delete(row_all, row_error)

        return df_date, df_day_96, row_nan_indices, row_zero_indices, row_error, row_all_right

    def deal_weather_data(self):
        df_weather = self.df_weather
        df_occupy = self.df_date.to_frame(name=0)
        df_occupy[1] = 0
        df_occupy[2] = np.nan

        df_weather = pd.concat([df_weather, df_occupy], ignore_index=True)

        df_weather_pivot = df_weather.pivot(index=0, columns=1, values=2)

        # wash data
        df_weather_pivot.loc[df_weather_pivot.loc[:,'最低温度'] < -100, ('最低温度', '平均温度')] = np.nan
        df_weather_pivot.loc[df_weather_pivot.loc[:, '湿度'] < 5, '湿度'] = np.nan
        df_weather_pivot.loc[df_weather_pivot.loc[:, '湿度'] > 100, '湿度'] = np.nan

        df_weather_pivot.loc[(df_weather_pivot.loc[:, '最低温度'] == 0) & (df_weather_pivot.loc[:, '最高温度'] == 0),
                ('最低温度', '平均温度', '最高温度')] = np.nan

        df_weather_pivot = df_weather_pivot.drop(0, axis=1)

        return df_weather_pivot

    def deal_time_onehot(self):
        date = self.df_date

        date_str = date.astype(str)
        date_pd = pd.to_datetime(date_str, format='%Y%m%d')
        date_li = date_pd.tolist()

        date_obj = [datetime.datetime.strptime(date_str_i, "%Y%m%d").date() for date_str_i in date_str]
        weekdays = [date_obj.weekday() for date_obj in date_obj]

        weekday = date_pd.dt.dayofweek

        cn_holiday = holidays.CountryHoliday('CN')
        holiday_info = [date_i in cn_holiday for date_i in date_pd]
        holiday_info_else_tu = [chinese_calendar.get_holiday_detail(date_i) for date_i in date_pd if date_i.year >= 2004]
        holiday_info_else = [if_holiday[0] for if_holiday in holiday_info_else_tu]

        weekend_and_holiday = []
        for i in range(len(holiday_info)):
            if i >= 365:
                if holiday_info[i] or holiday_info_else[i-365]:
                    weekend_and_holiday.append(i)
            else:
                if holiday_info[i] or weekday[i] == 5 or weekday[i] == 6:
                    weekend_and_holiday.append(i)

        weekend_and_holiday_onehot = np.zeros(len(date))
        weekend_and_holiday_onehot[weekend_and_holiday] = 1

        weekend = np.where(weekday.isin([5,6]))[0].tolist()
        weekend_onehot = np.zeros(len(date))
        weekend_onehot[weekend] = 1

        spring_fes_onehot = [Data.during_lunar_new_year(date_i) for date_i in date_li]
        spring_fes = np.where(np.array(spring_fes_onehot) >= 1)[0].tolist()

        # 提取年份、月份和星期
        df = date_pd.to_frame(name='date')

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday

        df_one_hot = pd.get_dummies(df, columns=['year', 'month', 'weekday'])
        date_one_hot = df_one_hot.iloc[:, 1:].to_numpy().astype(int)

        return (weekend, weekend_onehot, weekend_and_holiday, weekend_and_holiday_onehot, spring_fes, spring_fes_onehot,
                date_one_hot, date_pd)

    @ staticmethod
    def deal_pu(data, row_error, type='mean'):
        # data = np.delete(data, row_error, axis=0)
        if type == 'mean':
            data_base = np.mean(data, axis=1)
            data_pu = (data.T / data_base).T
        elif type == 'max':
            data_base = np.max(data, axis=1)
            data_pu = (data.T / data_base).T
        elif type == 'min':
            data_base = np.min(data, axis=1)
            data_pu = (data.T / data_base).T
        else:
            data_base = np.mean(data, axis=1)
            data_pu = (data.T / data_base).T

        return data_base, data_pu

    @ staticmethod
    def during_lunar_new_year(target_date):
        lunar_date = lunardate.LunarDate.fromSolarDate(target_date.year, target_date.month, target_date.day)
        if lunar_date.month == 1 and lunar_date.day <= 3:
            return 2
        elif lunar_date.month == 1 and lunar_date.day <= 6:
            return 1
        elif lunar_date.month == 12 and lunar_date.day >= 29:
            return 2
        elif lunar_date.month == 12 and lunar_date.day >= 28:
            return 1
        else:
            return 0

if __name__ == "__main__":
    data = Data('STLF_DATA_IN_1.xls')
    data.deal_pu(type='mean')
    pass


