import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#BTC data
file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-1d.csv'
df = pd.read_csv(file)
df['Datetime'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
# convert the column (it's a string) to datetime type
datetime_series = pd.to_datetime(df['Datetime'])
# create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(datetime_series.values)
df = df.set_index(datetime_index)
# we don't need the column anymore
df.drop('Datetime', axis=1, inplace=True)
df = df.sort_index()

profit_year = (df.groupby(pd.DatetimeIndex(df.index).to_period('Y')).Close.nth([-1])-df.groupby(pd.DatetimeIndex(df.index).to_period('Y')).Close.nth([0]))/df.groupby(pd.DatetimeIndex(df.index).to_period('Y')).Close.nth([0])*100

for year in profit_year.index:
    print(f'------ {year} ------')
    print(f'Return [%]: {round(profit_year[profit_year.index == year].values[0],2)} %')

colors = pd.DataFrame(profit_year.values>0).replace(True,'green').replace(False, 'red')[0]
plt.figure()
plt.bar(profit_year.index.to_series().astype(str), profit_year.values, color=list(colors))
plt.title('Evolution Bitcoin returns')
plt.ylabel('Returns [%]')
plt.show()