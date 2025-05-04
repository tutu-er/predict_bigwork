import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from data import Data

data = Data('STLF_DATA_IN_1.xls')

data_pph = {
    'ds': data.date_info.iloc[data.row_all_right],  # 5年每日数据
    'y': np.mean(data.np_day_pd_96[data.row_all_right, :], axis=1)
}
df = pd.DataFrame(data_pph)

model = Prophet(
    seasonality_mode='multiplicative',  # 季节性模式（加法或乘法）
    yearly_seasonality=True,            # 启用年季节性
    weekly_seasonality=True,            # 启用周季节性
    daily_seasonality=False
)
model.add_country_holidays(country_name='CN')

model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 绘制预测结果
fig1 = model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

# 分解趋势和季节性
fig2 = model.plot_components(forecast)
plt.show()

pass