import pandas as pd
import numpy as np
import talib
import time
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn import pipeline
from sklearn import metrics

import torch
import torch.nn as nn

def create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-4h.csv', period='4H', output = 10):
    # -------- DETERMINE THE PERIOD -------- https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    time_14 = 14
    time_25 = 25
    time_150 = 150
    # -------------------------------------

    df = pd.read_csv(file)
    df['Datetime'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0))
    col_study = ['Datetime', 'Close', 'Volume', 'NumberOfTrades']
    df = df[col_study]

    # convert the column (it's a string) to datetime type
    datetime_series = pd.to_datetime(df['Datetime'])
    # create datetime index passing the datetime series
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df = df.set_index(datetime_index)
    # we don't need the column anymore
    df.drop('Datetime',axis=1,inplace=True)
    df = df.sort_index()

    # Realized volatility last 10 periods
    df['RV'] = (np.log(df['Close']) - np.log(df['Close'].shift(10)))**2

    # Volume aggregation
    df['Volume_sum'] = df['Volume'].rolling(period).sum()

    # Number trades
    df['NTrades_sum'] = df['NumberOfTrades'].rolling(period).sum()

    # EMA - Exponential Moving Average
    df['EMA14'] = talib.EMA(df.Close, timeperiod=time_14)
    df['EMA25'] = talib.EMA(df.Close, timeperiod=time_25)
    df['EMA150'] = talib.EMA(df.Close, timeperiod=time_150)

    df['Dist_EMA14'] = (df.Close - df['EMA14'])/df.Close*100
    df['Dist_EMA25'] = (df.Close - df['EMA25'])/df.Close*100
    df['Dist_EMA150'] = (df.Close - df['EMA150'])/df.Close*100

    #Calculate output
    df['Output'] = df['Close'].shift(-output)

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)

    #Order final
    #col_study = ['Output', 'Close', 'RV', 'Volume_sum', 'NTrades_sum', 'Dist_EMA14', 'Dist_EMA25', 'Dist_EMA150']
    col_study = ['Output', 'Close']
    df = df[col_study]

    # Split train / test
    df_train = df[df.index.year < 2022]
    df_test = df[df.index.year >= 2022]

    #Scale data
    scaler = StandardScaler()
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train))
    df_test_scaled = pd.DataFrame(scaler.transform(df_test))

    x_train,y_train = df_train_scaled.iloc[:,1:], df_train_scaled.iloc[:,0]
    x_test, y_test = df_test_scaled.iloc[:, 1:], df_test_scaled.iloc[:, 0]

    return x_train, y_train, x_test, y_test

def ML_train(x_train, y_train, x_test, y_test):


    tree_regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "AdaBoost": AdaBoostRegressor(n_estimators=100),
        "Skl GBM": GradientBoostingRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=100),
        "LightGBM": LGBMRegressor(n_estimators=200),
        "CatBoost": CatBoostRegressor(n_estimators=100),
    }

    results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], 'Time': []})

    for model_name, model in tree_regressors.items():
        start_time = time.time()
        model.fit(x_train, y_train)
        total_time = time.time() - start_time

        pred = model.predict(x_test)

        results = pd.concat([results, pd.DataFrame({"Model": model_name,
                                  "MSE": metrics.mean_squared_error(y_test, pred),
                                  "MAB": metrics.mean_absolute_error(y_test, pred),
                                  "Time": total_time}, index=[0])],
                                 sort=False)

    results.reset_index(drop=True, inplace=True)
    results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
    results_ord.index += 1
    results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

    # Graph train result
    train_pred = tree_regressors[results['Model'][0]].predict(x_train)
    plt.plot(y_train, label="Real", linewidth=3)
    plt.plot(train_pred, label="Prediction", linewidth=1)
    plt.legend()
    plt.show()

    # Graph test prediction
    best_pred = tree_regressors[results['Model'][0]].predict(x_test)
    plt.plot(y_test, label="Real",linewidth=3)
    plt.plot(best_pred, label="Prediction",linewidth=1)
    plt.legend()
    plt.show()
    print(results_ord)

    return tree_regressors

if __name__=='__main__':

    x_train, y_train, x_test, y_test = create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-4h.csv', period='4H', output = 10)
    models = ML_train(x_train, y_train, x_test, y_test)

    print('done')