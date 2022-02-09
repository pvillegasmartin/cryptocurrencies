import os
import general_settings

import pandas as pd
import numpy as np
import talib

from sklearn.preprocessing import StandardScaler


def read_data(file):
    """Get Historical data from files
    :param file: Name of symbol pair e.g BNBBTC-hist
    :return: dataframe of OHLCV values
    """

    file_path = os.path.join(general_settings.path, 'get_data', file)

    # read csv
    df = pd.read_csv(file_path, header=0, sep=',', quotechar='"')

    # TODO which columns?
    # select only interesting columns
    df = df.loc[:, ['time', 'Open', 'Close','High', 'Low']]

    return df

def create_indicators(df):
    """Calculate financial indicators. Library documents: https://mrjbq7.github.io/ta-lib/doc_index.html
        :param df: Dataframe with OHLCV data
        :return: dataframe of OHLCV + financial indicators
    """
    # TODO which volatility indicators?
    # ----- Volatility Indicator -----
    # NATR - Normalized Average True Range
    natr = talib.NATR(df.High, df.Low, df.Close, timeperiod=14)
    df['NATR'] = natr
    del natr

    # ----- Sum changes '1m' -----
    df_1m = read_data('BTCUSDT-1m.csv')
    df_1m['HL'] = df_1m['High'] - df_1m['Low']
    df_1m['total_change'] = df_1m['HL'].rolling(min_periods=1, window=15).sum()
    df_1m = df_1m.loc[:,['time','total_change']]
    df = df.merge(df_1m, left_on='time', right_on='time')
    del df_1m
    # Normalize change of volume with average price
    df['avg_price'] = talib.AVGPRICE(df.Open, df.High, df.Low, df.Close)
    df['total_change_norm'] = df['total_change'] / df['avg_price']

    # Delete columns we don't need anymore
    df = df.loc[:,['NATR','total_change_norm']]

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

    # Scale numeric data
    scaler = StandardScaler()
    df_train_scaled = scaler.fit_transform(df_train)

    return df_train_scaled, df_test, scaler