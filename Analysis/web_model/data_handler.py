import pandas as pd
import numpy as np
import talib
import time
import matplotlib.pyplot as plt
import datetime
from scipy.signal import argrelextrema

import sklearn as skl

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import linear_model    # LogisticRegression
from sklearn import set_config

import settings as st


def create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-1d.csv'):
    df = pd.read_csv(file)
    df['Datetime'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    col_study = ['Datetime', 'Close', 'High', 'Low', 'Open']
    df = df[col_study]

    df.drop_duplicates(subset=['Datetime'], inplace=True)
    # DELETE
    #df = df[df['Datetime'].dt.year > 2019]

    df.reset_index(inplace=True)

    # Important points
    df['max_local'] = df.apply(lambda row: df.Close[(df['index'] > row['index'] - st.local_extreme) & (
            df['index'] < row['index'] + st.local_extreme)].max() == row.Close, axis=1)
    df['min_local'] = df.apply(lambda row: df.Close[(df['index'] > row['index'] - st.local_extreme) & (
            df['index'] < row['index'] + st.local_extreme)].min() == row.Close, axis=1)

    df.loc[df['max_local'] == True, 'output'] = 1
    df.loc[df['min_local'] == True, 'output'] = -1
    df['output'] = df['output'].fillna(0)

    col_study = ['output', 'Close', 'High', 'Low', 'Open']
    df = df[col_study]
    df.dropna(inplace=True)

    return df

def preprocess(dataset):
    x = dataset.drop('output', axis=1)
    y = dataset['output']
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle=False)

    # scaler = preprocessing.MinMaxScaler()
    # x_train_prepro = scaler.fit_transform(x_train)
    # x_val_prepro = scaler.transform(x_val)

    return x_train, y_train, x_val, y_val

if __name__ == '__main__':
    data = create_data()
    x_train, y_train, x_val, y_val = preprocess(data)