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
    df['last_max'] = (df['max_local'] * df['index']).replace(0, np.nan).shift(1).fillna(method="ffill")
    df['last_min'] = (df['min_local'] * df['index']).replace(0, np.nan).shift(1).fillna(method="ffill")
    # AUX TAKING REMARKABLE POINTS
    df['next_max'] = (df['max_local'] * df['index']).replace(0, np.nan).shift(-1).fillna(method="bfill")
    df['next_min'] = (df['min_local'] * df['index']).replace(0, np.nan).shift(-1).fillna(method="bfill")
    df.dropna(inplace=True)
    df['value_last_max'] = df.apply(lambda row: df.Close[df['index'] == int(row['last_max'])].max(), axis=1)
    df['value_last_min'] = df.apply(lambda row: df.Close[df['index'] == int(row['last_min'])].max(), axis=1)
    df['value_next_max'] = df.apply(lambda row: df.Close[df['index'] == int(row['next_max'])].max(), axis=1)
    df['value_next_min'] = df.apply(lambda row: df.Close[df['index'] == int(row['next_min'])].max(), axis=1)

    # # Puntos inflexion
    # df['inflex_max_up'] = 0
    # df['inflex_min_up'] = 0
    # df['inflex_max_down'] = 0
    # df['inflex_min_down'] = 0
    #
    # for index in df['index']:
    #     last_min = df[df['index'] == index].last_min.values[0]
    #     last_max = df[df['index'] == index].last_max.values[0]
    #     next_min = df[df['index'] == index].next_min.values[0]
    #     next_max = df[df['index'] == index].next_max.values[0]
    #     if last_min > last_max and next_max < next_min:
    #         df.loc[df['index'] == index, 'inflex_max_up'] = df[(df['index'] > last_max+1) & (df['index'] < next_max-1)][
    #                                                             'Close'].max() == df[df['index'] == index].Close.values[
    #                                                             0]
    #         df.loc[df['index'] == index, 'inflex_min_up'] = df[(df['index'] > last_max+1) & (df['index'] < next_max-1)][
    #                                                             'Close'].max() == df[df['index'] == index].Close.values[
    #                                                             0]
    #     elif last_min < last_max and next_max > next_min:
    #         df.loc[df['index'] == index, 'inflex_max_down'] = df[(df['index'] > last_min+1) & (df['index'] < next_min-1)][
    #                                                               'Close'].max() == \
    #                                                           df[df['index'] == index].Close.values[0]
    #         df.loc[df['index'] == index, 'inflex_min_down'] = df[(df['index'] > last_min+1) & (df['index'] < next_min-1)][
    #                                                               'Close'].min() == \
    #                                                           df[df['index'] == index].Close.values[0]
    # df['max_inflex'] = df['inflex_max_up'] + df['inflex_max_down']
    # df['min_inflex'] = df['inflex_min_up'] + df['inflex_min_down']

    # Important points dataframe
    max_points = pd.DataFrame(
        {'index': df[df['max_local'] == True]['index'], 'Type': 'Max', 'Value': df[df['max_local'] == True]['Close']})
    min_points = pd.DataFrame(
        {'index': df[df['min_local'] == True]['index'], 'Type': 'Min', 'Value': df[df['min_local'] == True]['Close']})
    # max_inflex = pd.DataFrame({'index': df[df['max_inflex'] == True]['index'], 'Type': 'Max_inflex',
    #                            'Value': df[df['max_inflex'] == True]['Close']})
    # min_inflex = pd.DataFrame({'index': df[df['min_inflex'] == True]['index'], 'Type': 'Min_inflex',
    #                            'Value': df[df['min_inflex'] == True]['Close']})
    important_points = pd.concat([max_points, min_points]) #, max_inflex, min_inflex])
    important_points = important_points.sort_values(by='index').reset_index(drop=True)

    # Create ML dataframe
    data_ML = pd.DataFrame(
        columns=['index', 'output', 'type_1', 'dif_1', 'timedif_1', 'type_2', 'dif_2', 'timedif_2', 'type_3', 'dif_3',
                 'timedif_3', 'type_4', 'dif_4', 'timedif_4', 'type_5', 'dif_5', 'timedif_5'])
    n_steps = st.n_steps
    first_point = important_points.iloc[n_steps - 1, 0]
    last_point = important_points.iloc[-1, 0]
    for row in df[(df['index'] > first_point+st.local_extreme) & (df['index'] < last_point)]['index']:
        row_close = df[df['index'] == row]['Close'].values[0]
        df_rel = important_points[important_points['index'] < row-st.local_extreme].iloc[-n_steps:, :]
        df_out = important_points[important_points['index'] > row].iloc[0, :]
        data_to_append = {'index': [row],
                          'output': [(df_out['Value'] - row_close) / row_close * 100],
                          'type_1': [df_rel.iloc[n_steps - 1, 1]],
                          'dif_1': [(row_close - df_rel.iloc[n_steps - 1, 2]) / df_rel.iloc[n_steps - 1, 2] * 100],
                          'timedif_1': [row - df_rel.iloc[n_steps - 1, 0]],
                          'type_2': [df_rel.iloc[n_steps - 2, 1]],
                          'dif_2': [(row_close - df_rel.iloc[n_steps - 2, 2]) / df_rel.iloc[n_steps - 2, 2] * 100],
                          'timedif_2': [row - df_rel.iloc[n_steps - 2, 0]],
                          'type_3': [df_rel.iloc[n_steps - 3, 1]],
                          'dif_3': [(row_close - df_rel.iloc[n_steps - 3, 2]) / df_rel.iloc[n_steps - 3, 2] * 100],
                          'timedif_3': [row - df_rel.iloc[n_steps - 3, 0]],
                          'type_4': [df_rel.iloc[n_steps - 4, 1]],
                          'dif_4': [(row_close - df_rel.iloc[n_steps - 4, 2]) / df_rel.iloc[n_steps - 4, 2] * 100],
                          'timedif_4': [row - df_rel.iloc[n_steps - 4, 0]],
                          'type_5': [df_rel.iloc[n_steps - 5, 1]],
                          'dif_5': [(row_close - df_rel.iloc[n_steps - 5, 2]) / df_rel.iloc[n_steps - 5, 2] * 100],
                          'timedif_5': [row - df_rel.iloc[n_steps - 5, 0]]}
        data_ML = pd.concat([data_ML, pd.DataFrame.from_dict(data_to_append)])

    return df, important_points, data_ML

def preprocess(dataset):
    dataset.drop('index', inplace=True, axis=1)
    x = dataset.drop('output', axis=1)
    y = dataset['output']
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle=False)
    num_vars = []
    cat_vars = []
    for i in range(1,st.n_steps+1):
        cat_vars.append(f'type_{i}')
        num_vars.append(f'dif_{i}')
        num_vars.append(f'timedif_{i}')

    num_preprocessing = pipeline.Pipeline(steps=[
        ('scaler', preprocessing.StandardScaler())
    ])

    cat_preporcessing = pipeline.Pipeline(steps=[
        ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))
    ])

    prepro = compose.ColumnTransformer(transformers=[
        ('num', num_preprocessing, num_vars),
        ('cat', cat_preporcessing, cat_vars),
    ], remainder='drop')

    x_train_prepro = prepro.fit_transform(x_train)
    x_val_prepro = prepro.transform(x_val)

    return x_train_prepro, y_train, x_val_prepro, y_val

if __name__ == '__main__':
    data, important_points, data_ML = create_data()
    x_train, y_train, x_val, y_val = preprocess(data_ML)
    # data.reset_index(inplace=True)
    # data['Close'].plot()
    # plt.scatter(data.index, data['max_local'] * data['Close'], color='green', alpha=0.5)
    # plt.scatter(data.index, data['min_local'] * data['Close'], color='orange', alpha=0.5)
    # # plt.scatter(data.index, data['max_inflex'] * data['Close'], color='red', alpha=0.5)
    # # plt.scatter(data.index, data['min_inflex'] * data['Close'], color='red', alpha=0.5)
    # plt.show()
    #
    # plt.hist(important_points[(important_points['Value'] > 20000) & (
    #         (important_points['Type'] == 'Max') | (important_points['Type'] == 'Min'))]['Value'], bins=100)
