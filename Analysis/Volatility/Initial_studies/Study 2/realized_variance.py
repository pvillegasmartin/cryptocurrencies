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

def create_indicators():
    """Calculate financial indicators. Library documents: https://mrjbq7.github.io/ta-lib/doc_index.html
        :param df: Dataframe with OHLCV data
        :return: dataframe of OHLCV + financial indicators
    """
    # ----- Volatility Indicator -----
    # ----- Realized variance -----
    df_1m = read_data('/00Versions/get_data/BTCUSDT-1m.csv')
    df_15m = read_data('/00Versions/get_data/BTCUSDT-15m.csv')

    # R2 10 mins window
    df_1m['10m'] = df_1m['Close'].shift(10)
    df_1m['R2_10m'] = (np.log(df_1m['Close']) - np.log(df_1m['10m']))**2
    weights = np.array(([0] * 9 + [1]) * 6 * 24)
    df_1m['R2_10'] = df_1m['R2_10m'].rolling(10 * 6 * 24).apply(lambda x: np.sum(weights * x))


    # R2 15 mins window
    df_1m['15m'] = df_1m['Close'].shift(15)
    df_1m['R2_15m'] = (np.log(df_1m['Close']) - np.log(df_1m['15m'])) ** 2
    weights = np.array(([0] * 14 + [1]) * 4 * 24)
    df_1m['R2_15'] = df_1m['R2_15m'].rolling(15 * 4 * 24).apply(lambda x: np.sum(weights * x))


    # R2 30 mins window
    df_1m['30m'] = df_1m['Close'].shift(30)
    df_1m['R2_30m'] = (np.log(df_1m['Close']) - np.log(df_1m['30m'])) ** 2
    weights = np.array(([0] * 29 + [1]) * 2 * 24)
    df_1m['R2_30'] = df_1m['R2_30m'].rolling(30 * 2 * 24).apply(lambda x: np.sum(weights * x))

    # R2
    df_1m['RV_1d'] = (df_1m['R2_10']+df_1m['R2_15']+df_1m['R2_30'])/3

    df_1m = df_1m.loc[:,['time','R2_10', 'R2_15', 'R2_30', 'RV_1d']]
    df = df_15m.merge(df_1m, how='left', on='time')
    del df_1m, df_15m

    # HAR
    weights = np.array(([0] * (24 * 4 - 1) + [1]) * 7)
    df['RV_7periods'] = df['RV_1d'].rolling(24*4 * 7).apply(lambda x: np.sum(weights * x)/7)
    weights = np.array(([0] * (24 * 4 - 1) + [1]) * 30)
    df['RV_30periods'] = df['RV_1d'].rolling(24 * 4 * 30).apply(lambda x: np.sum(weights * x)/30)

    # Output
    df['output'] = df['RV_1d'].shift(-24*4)

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save df preprocessed
    df.to_csv('volatility_desglose.csv', index=False)

    return df

def ML_prepare(df):
    """Final step previous to trainning model.
            :param df: Dataframe with volatility data
            :return: dataframe scaled and divided train/test. It returns the scaler to use it in future steps.
    """
    # TODO how many years for training/testing?
    # Divide dataframe in train/test
    df_train, df_test = np.split(df, [int(.75 * len(df))])
    df_test.reset_index(drop=True, inplace=True)

    axis_x_train = df_train['time'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0))
    axis_x_test = df_test['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))

    x_train, y_train = df_train.loc[:,['RV_1d', 'RV_7periods', 'RV_30periods']], df_train['output']
    x_test, y_test = df_test.loc[:, ['RV_1d', 'RV_7periods', 'RV_30periods']], df_test['output']

    # Scale numeric data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler, axis_x_train, axis_x_test

def ML_train(x_train, y_train, x_test, y_test, axis_x_train, axis_x_test):


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
    plt.plot(axis_x_train, y_train, label="Real", linewidth=3)
    plt.plot(axis_x_train, train_pred, label="Prediction", linewidth=1)
    plt.legend()
    plt.show()

    # Graph best prediction
    best_pred = tree_regressors[results['Model'][0]].predict(x_test)
    plt.plot(axis_x_test, y_test, label="Real",linewidth=3)
    plt.plot(axis_x_test, best_pred, label="Prediction",linewidth=1)
    plt.legend()
    plt.show()
    print(results_ord)

    return tree_regressors

def DL_split_data(df, lookback):

    df = df.loc[:, ['RV_1d', 'output']]
    data_raw = df.to_numpy()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)


    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :, :-1]
    y_train = data[:train_set_size, :, -1]

    x_test = data[train_set_size:, :, :-1]
    y_test = data[train_set_size:, :, -1]

    return x_train, y_train, x_test, y_test

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

def DL_train(df):

    # Number of steps to unroll
    seq_dim = 30

    # DATA
    x_train, y_train, x_test, y_test = DL_split_data(df, seq_dim)
    x_train, y_train, x_test, y_test = x_train.astype('float64'), y_train.astype('float64'), x_test.astype(
        'float64'), y_test.astype('float64')

    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_train, y_train = x_train.float().requires_grad_(), y_train.float()

    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
    x_test, y_test = x_test.float().requires_grad_(), y_test.float()

    # MODEL
    n_iters = 150
    # Number of features
    input_dim = x_train.shape[-1]
    hidden_dim = 100
    layer_dim = 1
    # Number of outputs
    output_dim = 1

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss(reduction='mean')

    iter = 0
    train_loss = []
    test_loss = []
    for it in range(n_iters):

        model.train()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        outputs = model(x_train)

        loss = criterion(outputs, y_train)
        print("Iteration ", it, "MSE: ", loss.item())

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 5 == 0:

            # Append the loss value
            train_loss.append(loss.item())

            with torch.no_grad():
                model.eval()

                y_test_pred = model(x_test)
                loss_eval = criterion(y_test_pred, y_test)
                # Append the loss value
                test_loss.append(loss.item())
                print("Iteration_test ", it, "MSE: ", loss_eval.item())

    #Graph error
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel(" Iteration ")
    plt.ylabel("Loss value")
    plt.legend(loc="upper left")
    plt.show()

    #Graph results
    model.eval()
    pred = model(x_test).detach().numpy()
    real = y_test[:,-1].detach().numpy()
    plt.plot(real, label="Real", linewidth=3)
    plt.plot(pred, label="Prediction", linewidth=1)
    plt.legend()
    plt.show()

    return model, x_test, y_test

if __name__=='__main__':
    #df_1d = create_indicators()
    """
    df_1d = pd.read_csv('volatility_desglose.csv', header=0)
    # Output
    df_1d['output'] = df_1d['RV_1d'].shift(-4 * 24)
    df_1d.dropna(inplace=True)
    x_train, y_train, x_test, y_test, scaler, axis_x_train, axis_x_test = ML_prepare(df_1d)
    models = ML_train(x_train, y_train, x_test, y_test, axis_x_train, axis_x_test)

    print('done')
    """
    df = read_data('C:/Users/Pablo/Desktop/STRIVE/cryptocurrencies/00Versions/get_data/BTCUSDT-1d.csv')
    df_1d = pd.read_csv('volatility_desglose.csv',header=0)
    df_1d = df_1d.loc[:,['time','RV_1d', 'output']]
    df_lstm = df.merge(df_1d, on='time', how='left')
    df_lstm.dropna(inplace=True)
    LSTM = DL_train(df_lstm)
