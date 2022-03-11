import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import talib

period = '4H'

def create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-4h.csv', kind='+', output=10,
                periods_out=6):

    df = pd.read_csv(file)
    df['Datetime'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0))
    col_study = ['Datetime', 'Close', 'Open', 'High', 'Low']
    df = df[col_study]

    # convert the column (it's a string) to datetime type
    datetime_series = pd.to_datetime(df['Datetime'])
    # create datetime index passing the datetime series
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df = df.set_index(datetime_index)
    # we don't need the column anymore
    df.drop('Datetime',axis=1,inplace=True)
    df = df.sort_index()

    # Calculate output
    if kind == '+':
        # POSITIVE
        df['Output_aux'] = df['Close'].shift(-periods_out).rolling(periods_out).max()
        df['Output'] = ((df['Output_aux'] - df['Close']) / df['Close']) > (output / 100)
        df['Output'] = df['Output'].map({True: 1, False: 0})
    else:
        # NEGATIVE
        df['Output_aux'] = df['Close'].shift(-periods_out).rolling(periods_out).min()
        df['Output'] = ((df['Output_aux'] - df['Close']) / df['Close']) < (-output / 100)
        df['Output'] = df['Output'].map({True: 1, False: 0})

    # Del Output_aux
    df.drop('Output_aux', axis=1, inplace=True)

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)

    # Split train / test
    df_train = df[df.index.year < 2022]
    df_test = df[df.index.year >= 2022]

    #Scale data
    scaler = StandardScaler()
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train.iloc[:,:-1]))
    df_test_scaled = pd.DataFrame(scaler.transform(df_test.iloc[:,:-1]))
    df_train_scaled['Output'] = df_train['Output'].values
    df_test_scaled['Output'] = df_test['Output'].values

    return df_train_scaled, df_test_scaled, scaler, df_train, df_test


def next_stock_batch(batch_size, df_base, n_steps=15, starting_points=None):
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
        for j in np.arange(n_steps+1):
            lmat.append(df_base.iloc[starting_points[k]+j,:].values)
            mat = np.array(lmat)
        # The x values include all columns (mat[:n_steps,:])
        # and TS values in mat between 0 and n_steps
        x[k,:,:] = mat[:n_steps,:-1]
        
        # The y values include only column 0 (mat[1:n_steps+1,0])
        # and TS values in mat between 1 and n_steps+1
        y[k,:,0] = mat[:n_steps,-1]

    return x,y
