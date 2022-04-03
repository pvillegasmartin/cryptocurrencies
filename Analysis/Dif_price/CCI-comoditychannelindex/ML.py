import pandas as pd
import numpy as np
import talib
import time
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from graphviz import Source
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier, plot_tree
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import confusion_matrix
from sklearn import pipeline
from sklearn import metrics

import pickle


# Commodity Channel Index
def CCI(df, ndays):
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['sma'] = df['TP'].rolling(ndays).mean()
    df['mad'] = df['TP'].rolling(ndays).apply(lambda x: pd.Series(x).mad())
    df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])
    return df


def create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-4h.csv'):
    df = pd.read_csv(file)
    df['Datetime'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    col_study = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df[col_study]

    # convert the column (it's a string) to datetime type
    datetime_series = pd.to_datetime(df['Datetime'])
    # create datetime index passing the datetime series
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df = df.set_index(datetime_index)
    # we don't need the column anymore
    df.drop('Datetime', axis=1, inplace=True)
    df = df.sort_index()

    # Filter first years
    df = df[df.index.year > 2017]

    # CCI
    df = CCI(df, 40)
    df['CCI-1'] = df.CCI.shift(1)

    # EMA - Exponential Moving Average
    df['EMA18'] = talib.EMA(df.Close, timeperiod=18)
    df['EMA200'] = talib.EMA(df.Close, timeperiod=200)

    # Trades
    df['index_aux'] = np.arange(len(df))
    df['long'] = (((df['CCI-1'] < -100) & (df['CCI'] > -100)) | ((df['CCI-1'] < 100) & (df['CCI'] > 100))) & (
            df['Open'] > \
            df['EMA18']) & (df['Open'] > df['EMA200']) & (df['EMA200'] > df['EMA200'].shift(1)) & (
                             df['EMA18'] > df['EMA18'].shift(1))
    df['sell'] = (((df['CCI-1'] > -100) & (df['CCI'] < -100)) | ((df['CCI-1'] > 100) & (df['CCI'] < 100))) & (
            df['Open'] < \
            df['EMA18']) & (df['Open'] < df['EMA200']) & (df['EMA200'] < df['EMA200'].shift(1)) & (
                             df['EMA18'] < df['EMA18'].shift(1))

    # Adaptation: no more than one buy or sell action at same time
    df['previous_long'] = df['index_aux'] * df['long']
    df['previous_long'] = df['previous_long'].replace(0, np.nan).fillna(method='ffill').shift(1)
    df['previous_sell'] = df['index_aux'] * df['sell']
    df['previous_sell'] = df['previous_sell'].replace(0, np.nan).fillna(method='ffill').shift(1)
    df['long'] = (df['long']) & (df['previous_sell'] > df['previous_long'])
    df['sell'] = (df['sell']) & (df['previous_sell'] < df['previous_long'])

    # Calculation of values when an order is made (used Close price)
    df['long_value'] = df['long'] * df['Close']
    df['sell_value'] = df['sell'] * df['Close']
    # Adaptation: if you cross the EMA200 close the position
    # df['sell_long_value'] = ((df.Close.shift(1) > df['EMA200']) & ((df.Close < df['EMA200']))) * df['EMA200']
    # df['sell_long_value'] = df['sell_long_value'].replace(0, np.nan).fillna(method='bfill')
    # df['sell_short_value'] = ((df.Close.shift(1) < df['EMA200']) & ((df.Close > df['EMA200']))) * df['EMA200']
    # df['sell_short_value'] = df['sell_short_value'].replace(0, np.nan).fillna(method='bfill')
    # Other option: close it when the opposite signal is present
    df['sell_short_value'] = df['long_value'].replace(0, np.nan).fillna(method='bfill')
    df['sell_long_value'] = df['sell_value'].replace(0, np.nan).fillna(method='bfill')

    # Output and results
    # df['output'] = ((df['EMA50'] - df['sell_long_value']) / df['EMA50'] * 100) > profit
    df['result_long'] = (df['sell_long_value'] - df['long_value']) / df['long_value'] * df['long'] * 100
    df['result_short'] = -(df['sell_short_value'] - df['sell_value']) / df['sell_value'] * df['sell'] * 100

    print(f"Buy and hold: {round((df.Close[-1]-df.Close[0])/df.Close[0]*100,2)} %")
    for year in set(df.index.year):
        print(f"    Buy and hold {year}: {round((df[df.index.year <= year].Close[-1] - df[df.index.year <= year].Close[0]) / df[df.index.year <= year].Close[0] * 100, 2)} %")

    print(f"Profit long: {round(df['result_long'].sum() - df['long'].sum() * 0.04,2)} %")
    for year in set(df.index.year):
        print(f"    Profit long {year}: {round(df[df.index.year <= year]['result_long'].sum() - df[df.index.year <= year]['long'].sum() * 0.04,2)} %")

    print(f"Profit short: {-round(df['result_short'].sum() + df['sell'].sum() * 0.04,2)} %")
    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)

    # EDA Plots
    '''
    # BTC Price and EMA
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(df['Close'], label='Close evolution', color='black', linewidth=1)
    ax1.plot(df['EMA18'], label='EMA', color='red', linewidth=0.5)
    ax1.plot(df['EMA200'], label='EMA', color='orange', linewidth=0.5)
    ax1.scatter(df[df['long'] == 1].index,
                df[df['long'] == 1]['Close'],
                color='green', alpha=0.8, label='buy')
    ax1.scatter(df[df['sell'] == 1].index,
                df[df['sell'] == 1]['Close'],
                color='red', alpha=0.8, label='sell')
    ax2.plot(df['CCI'], label='CCI', color='black', linewidth=1)
    ax2.scatter(df[df['long'] == 1].index,
                df[df['long'] == 1]['CCI'],
                color='green', alpha=0.8, label='buy')
    plt.show()
    '''

    return df


if __name__ == '__main__':
    df = create_data()
    print('mnak')
