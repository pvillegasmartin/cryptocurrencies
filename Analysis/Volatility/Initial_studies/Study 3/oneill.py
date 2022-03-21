#https://www.oreilly.com/library/view/machine-learning-for/9781492085249/ch04.html
import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
import yfinance as yf
import pandas as pd
import datetime
import time
from arch import arch_model
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error as mse
import warnings

from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')

stocks = '^GSPC'
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2021, 8, 1)
s_p500 = yf.download(stocks, start=start, end = end, interval='1d')

ret = 100 * (s_p500.pct_change()[1:]['Adj Close'])
realized_vol = ret.rolling(5).std()
'''
plt.figure(figsize=(10, 6))
plt.plot(realized_vol.index,realized_vol)
plt.title('Realized Volatility- S&P-500')
plt.ylabel('Volatility')
plt.xlabel('Date')
plt.show()

retv = ret.values
plt.figure(figsize=(10, 6))
plt.plot(s_p500.index[1:], ret)
plt.title('Volatility clustering of S&P-500')
plt.ylabel('Daily returns')
plt.xlabel('Date')
plt.show()
'''
returns_svm = ret ** 2
returns_svm = returns_svm.reset_index()
del returns_svm['Date']
X = pd.DataFrame(realized_vol)
X = X.reset_index(drop=True)
X['returns_svm'] = returns_svm.iloc[:,0]
X = X[4:].copy()
X = X.reset_index(drop=True)

n = 252

model = keras.Sequential(
             [layers.Dense(256, activation="relu"),
              layers.Dense(128, activation="relu"),
              layers.Dense(1, activation="linear"),])

model.compile(loss='mse', optimizer='rmsprop')

epochs_trial = np.arange(100, 400, 4)
batch_trial = np.arange(100, 400, 4)
DL_pred = []
DL_RMSE = []
for i, j, k in zip(range(4), epochs_trial, batch_trial):
     model.fit(X.iloc[:-n].values,
               realized_vol.iloc[5:-(n-1)].values.reshape(-1,),
               batch_size=k, epochs=j, verbose=False)
     DL_predict = model.predict(np.asarray(X.iloc[-n:]))
     DL_RMSE.append(np.sqrt(mse(realized_vol.iloc[-n:] / 100,
                             DL_predict.flatten() / 100)))
     DL_pred.append(DL_predict)
     print('DL_RMSE_{}:{:.6f}'.format(i+1, DL_RMSE[i]))

DL_predict = pd.DataFrame(DL_pred[DL_RMSE.index(min(DL_RMSE))])
DL_predict.index = ret.iloc[-n:].index
plt.figure(figsize=(10, 6))
plt.plot(realized_vol / 100,label='Realized Volatility')
plt.plot(DL_predict / 100,label='Volatility Prediction-DL')
plt.title('Volatility Prediction with Deep Learning',  fontsize=12)
plt.legend()
plt.show()