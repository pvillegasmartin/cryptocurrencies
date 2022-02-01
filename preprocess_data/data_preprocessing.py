import os
import general_settings

import pandas as pd
import talib


def read_data(file=os.path.join(general_settings.path,'get_data','BTCUSDT-hist.csv')):
    """Get Historical data from files
    :param file: Name of symbol pair e.g BNBBTC-hist
    :return: dataframe of OHLCV values
    """
    # read zip
    df = pd.read_csv(file, header=0, sep=',', quotechar='"')

    # TODO which columns?
    # select only interesting columns
    df = df.loc[:, ['Open', 'Close','High', 'Low']]

    return df

def create_indicators(df):
    """Calculate financial indicators. Library documents: https://mrjbq7.github.io/ta-lib/doc_index.html
        :param df: Dataframe with OHLCV data
        :return: dataframe of OHLCV + financial indicators
    """
    # TODO which momentum indicators?
    # ----- Momentum Indicator -----
    # MACD - Moving Average Convergence/Divergence
    macd, macdsignal, macdhist = talib.MACD(df.Close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd

    # TODO which patterns on candlesticks?
    # ----- Candlesticks Pattern Recognition -----
    # Dragonfly Doji
    df['dragonfly'] = talib.CDLDRAGONFLYDOJI(df.Open, df.High, df.Low, df.Close)

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# TODO implement scalers
def scaler(df):
    return df

if __name__ == '__main__':
    df = read_data()
    df = create_indicators(df)
    print('done')