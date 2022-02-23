import pandas as pd
import numpy as np
import talib
import time
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
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

import torch
import torch.nn as nn


def data_enhancement(data):
    gen_data = data[data['Output'] == True]

    Close_std = gen_data['Close'].std()
    RV_std = gen_data['RV'].std()
    Volume_sum_std = gen_data['Volume_sum'].std()
    NTrades_sum_std = gen_data['NTrades_sum'].std()
    Dist_EMA14_std = gen_data['Dist_EMA14'].std()
    Dist_EMA25_std = gen_data['Dist_EMA25'].std()
    Dist_EMA150_std = gen_data['Dist_EMA150'].std()

    for i in range(len(gen_data)):
        if np.random.randint(2) == 1:
            gen_data['Close'].values[i] += Close_std / 10
        else:
            gen_data['Close'].values[i] -= Close_std / 10

        if np.random.randint(2) == 1:
            gen_data['RV'].values[i] += RV_std / 10
        else:
            gen_data['RV'].values[i] -= RV_std / 10

        if np.random.randint(2) == 1:
            gen_data['Volume_sum'].values[i] += Volume_sum_std / 10
        else:
            gen_data['Volume_sum'].values[i] -= Volume_sum_std / 10

        if np.random.randint(2) == 1:
            gen_data['NTrades_sum'].values[i] += NTrades_sum_std / 10
        else:
            gen_data['NTrades_sum'].values[i] -= NTrades_sum_std / 10

        if np.random.randint(2) == 1:
            gen_data['Dist_EMA14'].values[i] += Dist_EMA14_std / 10
        else:
            gen_data['Dist_EMA14'].values[i] -= Dist_EMA14_std / 10

        if np.random.randint(2) == 1:
            gen_data['Dist_EMA25'].values[i] += Dist_EMA25_std / 10
        else:
            gen_data['Dist_EMA25'].values[i] -= Dist_EMA25_std / 10

        if np.random.randint(2) == 1:
            gen_data['Dist_EMA150'].values[i] += Dist_EMA150_std / 10
        else:
            gen_data['Dist_EMA150'].values[i] -= Dist_EMA150_std / 10

    return gen_data


def create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-4h.csv', period='4H', output=10,
                periods_out=6):
    # -------- DETERMINE THE PERIOD -------- https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    time_14 = 14
    time_25 = 25
    time_150 = 150
    # -------------------------------------

    df = pd.read_csv(file)
    df['Datetime'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    col_study = ['Datetime', 'Close', 'Volume', 'NumberOfTrades']
    df = df[col_study]

    # convert the column (it's a string) to datetime type
    datetime_series = pd.to_datetime(df['Datetime'])
    # create datetime index passing the datetime series
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df = df.set_index(datetime_index)
    # we don't need the column anymore
    df.drop('Datetime', axis=1, inplace=True)
    df = df.sort_index()

    # Realized volatility last 10 periods
    df['RV'] = (np.log(df['Close']) - np.log(df['Close'].shift(10))) ** 2

    # Volume aggregation
    df['Volume_sum'] = df['Volume'].rolling(period).sum()

    # Number trades
    df['NTrades_sum'] = df['NumberOfTrades'].rolling(period).sum()

    # EMA - Exponential Moving Average
    df['EMA14'] = talib.EMA(df.Close, timeperiod=time_14)
    df['EMA25'] = talib.EMA(df.Close, timeperiod=time_25)
    df['EMA150'] = talib.EMA(df.Close, timeperiod=time_150)

    df['Dist_EMA14'] = (df.Close - df['EMA14']) / df.Close * 100
    df['Dist_EMA25'] = (df.Close - df['EMA25']) / df.Close * 100
    df['Dist_EMA150'] = (df.Close - df['EMA150']) / df.Close * 100

    # Calculate output
    df['Output_aux'] = df['Close'].shift(-periods_out).rolling(periods_out).max()
    df['Output'] = abs(df['Output_aux'] - df['Close']) / df['Close'] > output / 100

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)

    # Order final
    col_study = ['Output', 'Close', 'RV', 'Volume_sum', 'NTrades_sum', 'Dist_EMA14', 'Dist_EMA25', 'Dist_EMA150']
    # col_study = ['Output', 'Close']
    df = df[col_study]

    # Data augmentation
    gen = data_enhancement(df)
    df = pd.concat([df, gen])

    # Split train / test
    y = df['Output']
    x = df.drop(['Output'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

    # Scale data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def ML_train(x_train, y_train, x_test, y_test):
    tree_classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "Skl GBM": GradientBoostingClassifier(n_estimators=100),
        "Skl HistGBM": HistGradientBoostingClassifier(max_iter=100),
        "XGBoost": XGBClassifier(n_estimators=100),
        "LightGBM": LGBMClassifier(n_estimators=100),
        #"CatBoost": CatBoostClassifier(n_estimators=100),
    }

    results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    for model_name, model in tree_classifiers.items():
        start_time = time.time()
        model.fit(x_train, y_train)
        total_time = time.time() - start_time

        pred = model.predict(x_test)

        results = results.append({"Model": model_name,
                                  "Accuracy": metrics.accuracy_score(y_test, pred) * 100,
                                  "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred) * 100,
                                  "Time": total_time},
                                 ignore_index=True)

    results.reset_index(drop=True, inplace=True)
    results_ord = results.sort_values(by=['Accuracy'], ascending=True, ignore_index=True)

    best_pred = tree_classifiers[results['Model'][0]].predict(x_test)
    print(confusion_matrix(y_test, best_pred))

    return tree_classifiers

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-4h.csv',
                                                   period='4H', output=10)
    models = ML_train(x_train, y_train, x_test, y_test)

    print('done')
