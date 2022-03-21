import pandas as pd
import numpy as np
import seaborn as sns
import talib
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats

def read_data(file):
    """Get Historical data from files
    :param file: Name of symbol pair e.g BNBBTC-hist
    :return: dataframe of OHLCV values
    """
    # read csv
    df = pd.read_csv(file, header=0, sep=',',
                         quotechar='"')

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
    df_1m = read_data('C:/Users/Pablo/Desktop/STRIVE/cryptocurrencies/00Versions/get_data/BTCUSDT-1m.csv')
    df_1m['HL'] = df_1m['High'] - df_1m['Low']
    df_1m['total_change'] = df_1m['HL'].rolling(min_periods=1, window=15).sum()
    df_1m = df_1m.loc[:,['time','total_change']]
    df = df.merge(df_1m, left_on='time', right_on='time')
    del df_1m
    # Normalize change of volume with average price
    df['avg_price'] = talib.AVGPRICE(df.Open, df.High, df.Low, df.Close)
    df['total_change_norm'] = df['total_change'] / df['avg_price']

    # Delete columns we don't need anymore
    #df = df.loc[:,['NATR','total_change_norm']]

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def stat_points(df, prob=0.21, dist='normal'):
    # We use the ppf function (inverse cdf) - from probability to z score
    df_mu = df.mean()
    df_sigma = df.std(ddof=0)
    p = 1 - prob
    z = stats.norm.ppf(p)
    # From z score to number of x
    x1 = df_mu + z * df_sigma
    x2 = df_mu - z * df_sigma
    real_value_1 = np.exp(x1) if dist=='lognormal' else x1
    real_value_2 = np.exp(x2) if dist=='lognormal' else x2
    df_mu = np.exp(df_mu) if dist == 'lognormal' else df_mu
    return df_mu, real_value_1, real_value_2

if __name__=='__main__':
    df_15 = read_data('C:/Users/Pablo/Desktop/STRIVE/cryptocurrencies/00Versions/get_data/BTCUSDT-15m.csv')
    df_15 = create_indicators(df_15)

    # TOTAL CHANGE NORM
    log_total_change = np.log(df_15[df_15['total_change_norm']>0]['total_change_norm'])
    '''
    f = Fitter(log_total_change,
               distributions=get_common_distributions())
    f.fit()
    print(f.summary())
    print(f.get_best(method='sumsquare_error'))
    '''
    prob = 0.21
    df_mu, real_value_1, real_value_2 = stat_points(log_total_change, prob, dist='lognormal')

    ax = sns.displot(df_15['total_change_norm'],
                      bins=100,
                      kde=True,
                      color='skyblue')
    ax.set(xlabel='Total_change Distribution', ylabel='Frequency')
    plt.axvline(real_value_1, color='red')
    plt.axvline(real_value_2, color='red')
    plt.axvline(df_mu, color='green')
    plt.annotate(str(prob*100) + '% - '+str(round(real_value_1,3)),(real_value_1,1000))
    plt.annotate(str(prob*100) + '% - '+str(round(real_value_2,3)),(0,0))
    plt.show()

    # NATR
    log_NATR = np.log(df_15[df_15['NATR'] > 0]['NATR'])
    '''
    f = Fitter(log_NATR,
               distributions=get_common_distributions())
    f.fit()
    print(f.summary())
    print(f.get_best(method='sumsquare_error'))
    
    prob=0.21
    df_mu, real_value_1, real_value_2 = stat_points(log_NATR, prob, dist='lognormal')

    ax = sns.displot(df_15['NATR'],
                     bins=100,
                     kde=True,
                     color='skyblue')
    ax.set(xlabel='NATR', ylabel='Frequency')
    plt.axvline(real_value_1, color='red')
    plt.axvline(real_value_2, color='red')
    plt.axvline(df_mu, color='green')
    plt.annotate(str(prob*100) + '% - '+str(round(real_value_1,3)),(real_value_1,0))
    plt.annotate(str(prob*100) + '% - '+str(round(real_value_2,3)),(0,0))
    plt.show()
    '''