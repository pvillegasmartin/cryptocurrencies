import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import talib

period = '4H'

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
    col_study = ['Output', 'Close', 'RV', 'Volume_sum', 'NTrades_sum', 'Dist_EMA14', 'Dist_EMA25', 'Dist_EMA150']
    #col_study = ['Output', 'Close']
    df = df[col_study]

    # Split train / test
    df_train = df[df.index.year < 2022]
    df_test = df[df.index.year >= 2022]

    #Scale data
    scaler = StandardScaler()
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train))
    df_test_scaled = pd.DataFrame(scaler.transform(df_test))

    return df_train_scaled, df_test_scaled, scaler, df_train, df_test


def next_stock_batch(batch_size, df_base, n_steps=7, starting_points=None):
    t_min = 0
    t_max = df_base.shape[0]
  
    # The inputs will be formed by X sequences
    x = np.zeros((batch_size,n_steps,len(df_base.columns)-1))
    
    # We want to predict the returns of the next
    y = np.zeros((batch_size,n_steps,1))

    if starting_points == None:
        # We chose batch_size random points from time series x-axis
        starting_points = np.random.randint(t_min, t_max-n_steps-1 ,size=batch_size)

    # We create the batches for x using all time series between t and t+n_steps with the corresponding jumps depending on period
    # We create the batches for y using only one time series between t+1 and t+n_steps+1 with the corresponding jumps depending on period
    
    for k in np.arange(batch_size):
        lmat = []
        for j in np.arange(n_steps):
            lmat.append(df_base.iloc[starting_points[k]+j,:].values)
            mat = np.array(lmat)
        # The x values include all columns (mat[:n_steps,:])
        # and TS values in mat between 0 and n_steps
        x[k,:,:] = mat[:n_steps,1:]
        
        # The y values include only column 0 (mat[1:n_steps+1,0])
        # and TS values in mat between 1 and n_steps+1
        y[k,:,0] = mat[:n_steps,0]

    return x,y
