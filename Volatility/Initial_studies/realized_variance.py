import pandas as pd
import numpy as np
import talib
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
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

def read_data(file):
    """Get Historical data from files
    :param file: Name of symbol pair e.g BNBBTC-hist
    :return: dataframe of OHLCV values
    """

    # read csv
    df = pd.read_csv(file, header=0, sep=',', quotechar='"')

    # TODO which columns?
    # select only interesting columns
    df = df.loc[:, ['time', 'Open', 'Close','High', 'Low']]

    return df

def create_indicators(df):
    """Calculate financial indicators. Library documents: https://mrjbq7.github.io/ta-lib/doc_index.html
        :param df: Dataframe with OHLCV data
        :return: dataframe of OHLCV + financial indicators
    """
    # ----- Volatility Indicator -----
    # ----- Realized variance -----
    df_1m = read_data('C:/Users/Pablo/Desktop/STRIVE/cryptocurrencies/00Versions/get_data/BTCUSDT-1m.csv')
    df_1m['R2'] = (np.log(df_1m['High']) - np.log(df_1m['Low']))**2
    # TODO periods to define rollings
    df_1m['RV_4h'] = df_1m['R2'].rolling(min_periods=1, window=4*60).sum()
    df_1m = df_1m.loc[:,['time','RV_4h']]
    df = df.merge(df_1m, left_on='time', right_on='time')
    del df_1m
    # Normalize change of volume with average price
    # df['avg_price'] = talib.AVGPRICE(df.Open, df.High, df.Low, df.Close)
    # df['total_change_norm'] = df['total_change'] / df['avg_price']

    # HAR
    df['RV_7periods'] = df['RV_4h'].rolling(min_periods=1, window=7).mean()
    df['RV_30periods'] = df['RV_4h'].rolling(min_periods=1, window=30).mean()

    # Output
    df['output'] = df['RV_4h'].shift(-1)

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def prepare(df):
    """Final step previous to trainning model.
            :param df: Dataframe with volatility data
            :return: dataframe scaled and divided train/test. It returns the scaler to use it in future steps.
    """
    # TODO how many years for training/testing?
    # Divide dataframe in train/test
    df_train, df_test = np.split(df, [int(.75 * len(df))])
    df_test.reset_index(drop=True, inplace=True)

    x_train, y_train = df_train.loc[:,['RV_4h', 'RV_7periods', 'RV_30periods']], df_train['output']
    x_test, y_test = df_test.loc[:, ['RV_4h', 'RV_7periods', 'RV_30periods']], df_test['output']

    # Scale numeric data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler

def train(x_train, y_train, x_test, y_test):

    rang = abs(y_train.max()) + abs(y_train.min())

    tree_regressors = {
        # "Decision Tree": DecisionTreeRegressor(),
        # "Extra Trees": ExtraTreesRegressor(n_estimators=100),
        # "Random Forest": RandomForestRegressor(n_estimators=100),
        # "AdaBoost": AdaBoostRegressor(n_estimators=100),
        # "Skl GBM": GradientBoostingRegressor(n_estimators=100),
        # "XGBoost": XGBRegressor(n_estimators=100),
        "LightGBM": LGBMRegressor(n_estimators=100),
        "CatBoost": CatBoostRegressor(n_estimators=100),
    }

    results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

    for model_name, model in tree_regressors.items():
        start_time = time.time()
        model.fit(x_train, y_train)
        total_time = time.time() - start_time

        pred = model.predict(x_test)

        results = results.append({"Model": model_name,
                                  "MSE": metrics.mean_squared_error(y_test, pred),
                                  "MAB": metrics.mean_absolute_error(y_test, pred),
                                  " % error": metrics.mean_squared_error(y_test, pred) / rang,
                                  "Time": total_time},
                                 ignore_index=True)

    results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
    results_ord.index += 1
    results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

    # Graph best result
    best_pred = tree_regressors[results['Model'][0]].predict(x_test)
    plt.plot(best_pred, label="Prediction")
    plt.plot(y_test, label="Real")
    plt.legend()
    plt.show()
    print(results_ord)

    return tree_regressors

if __name__=='__main__':
    df_4h = read_data('C:/Users/Pablo/Desktop/STRIVE/cryptocurrencies/00Versions/get_data/BTCUSDT-4h.csv')
    df_4h = create_indicators(df_4h)
    x_train, y_train, x_test, y_test, scaler = prepare(df_4h)
    models = train(x_train, y_train, x_test, y_test)


    print('a')

