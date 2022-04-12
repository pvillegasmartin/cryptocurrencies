import pandas as pd
import numpy as np
import talib
import time
import matplotlib.pyplot as plt
import datetime
from scipy.signal import argrelextrema


def create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-1d.csv'):
    df = pd.read_csv(file)
    df['Datetime'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    col_study = ['Datetime', 'Close', 'High', 'Low', 'Open']
    df = df[col_study]

    df.drop_duplicates(subset=['Datetime'], inplace=True)
    # DELETE
    #df = df[df['Datetime'].dt.year > 2019]

    df.reset_index(inplace=True)

    # Important points
    max_idx = list(argrelextrema(df.Close.values, np.greater, order=6)[0])
    min_idx = list(argrelextrema(df.Close.values, np.less, order=6)[0])

    plt.figure()
    plt.plot(df.Close)
    plt.scatter(max_idx, df.Close[max_idx], c='green')
    plt.scatter(min_idx, df.Close[min_idx], c='red')
    plt.show()

if __name__ == '__main__':
    create_data()