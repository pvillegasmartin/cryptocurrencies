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


def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr


def create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-15m.csv', profit=5):
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

    # Williams
    df['wr_21'] = get_wr(df['High'], df['Low'], df['Close'], 21)
    df['EMA13'] = talib.EMA(df.wr_21, timeperiod=13)

    # MACD
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df.Close, fastperiod=12, slowperiod=26, signalperiod=9)

    # EMA - Exponential Moving Average
    df['EMA50'] = talib.EMA(df.Close, timeperiod=50)

    # Previous values
    df['macdhist-1'], df['EMA50-1'], df['wr_21-1'], df['EMA13-1'], df['Close-1'] = df['macdhist'].shift(1), df[
        'EMA50'].shift(1), df['wr_21'].shift(1), df['EMA13'].shift(1), df['Close'].shift(1)
    df['macdhist-2'], df['EMA50-2'], df['wr_21-2'], df['EMA13-2'], df['Close-2'] = df['macdhist'].shift(2), df[
        'EMA50'].shift(2), df['wr_21'].shift(2), df['EMA13'].shift(2), df['Close'].shift(2)

    # Trades
    df['long_MACD'] = (df['macdhist'] > 0) & ((df['macdhist-1'] < 0) | (df['macdhist-2'] < 0))
    df['long_will'] = (df['wr_21'] > df['EMA13']) & ((df['wr_21-1'] < df['EMA13-1']) | (df['wr_21-2'] < df['EMA13-2']))
    df['long_EMA'] = (df.Close > df['EMA50']) & (df.Open < df['EMA50'])
    df['long'] = df['long_MACD'] & df['long_will'] & df['long_EMA']
    df['long_value'] = df['long'] * df['EMA50']
    df['sell'] = (df.Low < df['EMA50']) & (df.Open > df['EMA50'])
    df['sell_value'] = df['sell'] * df['EMA50']
    df['sell_value'] = df['sell_value'].replace(0, np.nan).fillna(method='bfill')
    df['output'] = ((df['EMA50'] - df['sell_value']) / df['EMA50'] * 100) > profit
    df['result'] = (df['sell_value'] - df['long_value']) / df['long_value'] * df['long'] * 100

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)

    # EDA Plots
    '''
    # BTC Price and EMA
    plt.plot(df['Close'], label='Close evolution', color='green', linewidth=1)
    plt.plot(df['EMA50'], label='Close evolution', color='black', linewidth=0.5)
    plt.scatter(df[df['long'] == 1].index,
                df[df['long'] == 1]['Close'],
                color='yellow', alpha=0.5, label='buy')
    plt.show()

    # All inputs graphs
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    #BTC Price and EMA
    ax1.plot(df['Close'], label='Close evolution', color='green', linewidth=1)
    ax1.plot(df['EMA50'], label='Close evolution', color='black', linewidth=0.5)
    ax1.scatter(df[df['long'] == 1].index,
               df[df['long'] == 1]['Close'],
               color='green', alpha=0.5, label='buy')
    # ax1.scatter(df[df['sell'] == 1].index,
    #             df[df['sell'] == 1]['Close'],
    #             color='red', alpha=0.5, label='sell')
    ax1.legend(loc='lower left')
    #MACD
    ax2.plot(df['macd'], label='macd', linewidth=1)
    ax2.plot(df['macdsignal'], label='macdsignal', linewidth=1)
    ax2.legend(loc='lower left')
    #Williams
    ax3.plot(df['wr_21'], label='William', linewidth=1)
    ax3.plot(df['EMA13'], label='William_ema', linewidth=1)
    ax3.legend(loc='lower left')
    plt.show()
    '''

    # Stay only with columns to do the study
    col_study = ['output', 'Open', 'Low', 'High', 'Close', 'EMA50', 'macdhist', 'macdhist-1', 'macdhist-2', 'wr_21',
                 'wr_21-1', 'wr_21-2', 'EMA13', 'EMA13-1', 'EMA13-2']
    df = df[col_study]
    x = df.drop('output', axis=1)
    y = df['output']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42,
                                                      shuffle=False)
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    return x_train_scaled, x_val_scaled, y_train, y_val

def ML_train(x_train, x_val, y_train, y_val, profit):
    tree_classifiers = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, bootstrap=True, class_weight='balanced_subsample'),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "Skl GBM": GradientBoostingClassifier(n_estimators=100),
        "Skl HistGBM": HistGradientBoostingClassifier(max_iter=100),
        "XGBoost": XGBClassifier(n_estimators=100),
        "LightGBM": LGBMClassifier(n_estimators=100),
        # "CatBoost": CatBoostClassifier(n_estimators=100),
    }

    results = pd.DataFrame({'Model': [], 'Precision': [], 'Time': []})

    for model_name, model in tree_classifiers.items():
        start_time = time.time()
        model.fit(x_train, y_train)
        total_time = time.time() - start_time

        pred = model.predict(x_val)

        results = results.append({"Model": model_name,
                                  "Precision": metrics.precision_score(y_val, pred) * 100,
                                  "Time": total_time},
                                 ignore_index=True)

    results.reset_index(drop=True, inplace=True)
    results_ord = results.sort_values(by=['Precision'], ascending=False, ignore_index=True)

    best_pred = tree_classifiers[results_ord['Model'][0]].predict(x_val)
    print(confusion_matrix(y_val, best_pred))

    # save the best model to disk
    filename = f'{results_ord["Model"][0]}_profit+{profit}.sav'
    pickle.dump(tree_classifiers[results_ord['Model'][0]], open(filename, 'wb'))

    return tree_classifiers


if __name__ == '__main__':
    profit = 3
    x_train, x_val, y_train, y_val = create_data(profit=profit)
    models = ML_train(x_train, x_val, y_train, y_val, profit=profit)
    y_pred = models['Extra Trees'].predict(x_val)
    print('mnak')
