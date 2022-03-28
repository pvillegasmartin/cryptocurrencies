import time
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

import matplotlib.pyplot as plt

import data_handler as dh
import settings as st

data, important_points, data_ML, first_point, last_point = dh.create_data()
x_train, y_train, x_val, y_val = dh.preprocess(data_ML)

classifiers = {
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "AdaBoost": AdaBoostRegressor(n_estimators=100),
    "Skl GBM": GradientBoostingRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100),
    "LightGBM": LGBMRegressor(n_estimators=100),
    "CatBoost": CatBoostRegressor(n_estimators=100)
}

results = pd.DataFrame({'Model': [], 'Explained_variance': [], 'MSE': [], 'MAE': [], "R2-score": [], 'Time': []})

for model_name, model in classifiers.items():
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time

    y_pred = model.predict(x_val)

    results = results.append({"Model": model_name,
                              "Explained_variance": metrics.explained_variance_score(y_val, y_pred),
                              "MSE": metrics.mean_squared_error(y_val, y_pred),
                              "MAE": metrics.mean_absolute_error(y_val, y_pred),
                              "R2-score": metrics.r2_score(y_val, y_pred),
                              "Time": total_time},
                             ignore_index=True)

results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)

# TESTING
y_pred = classifiers[results_ord['Model'][0]].predict(x_val)

data_to_plot = data[(data['index'] > first_point) & (data['index'] < last_point)].iloc[-y_val.shape[0] - 1:, :]
data_to_plot.reset_index(inplace=True, drop=True)

fig, ax = plt.subplots()
ax.plot(data_to_plot['Close'], label='Close evolution', color='green')
ax.scatter(data_to_plot[data_to_plot['max_local'] == 1].index, data_to_plot[data_to_plot['max_local'] == 1]['Close'],
           color='green', alpha=0.5, label='max')
ax.scatter(data_to_plot[data_to_plot['min_local'] == 1].index, data_to_plot[data_to_plot['min_local'] == 1]['Close'],
           color='orange', alpha=0.5, label='min')
ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.plot(np.linspace(1, y_val.shape[0], y_val.shape[0]), y_val, label='real', linewidth=1)
ax2.plot(np.linspace(1, y_val.shape[0], y_val.shape[0]), y_pred, linestyle='dashed', label='prediction', linewidth=0.5)
ax2.legend(loc='upper right')
plt.show()

print(
    f'Y_val summary --- Mean: {round(y_val.mean(), 2)}%  Max: {round(y_val.max(), 2)}%  Min: {round(y_val.min(), 2)}%')
print(results_ord.iloc[0])
