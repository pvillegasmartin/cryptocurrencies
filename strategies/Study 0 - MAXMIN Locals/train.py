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

data, important_points, data_ML = dh.create_data()
x_train, y_train, x_val, y_val = dh.preprocess(data_ML)

classifiers = {
  "SVR": SVR(),
  "Decision Tree": DecisionTreeRegressor(),
  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
  "Random Forest": RandomForestRegressor(n_estimators=100),
  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
  "XGBoost":       XGBRegressor(n_estimators=100),
  "LightGBM":      LGBMRegressor(n_estimators=100),
  "CatBoost":      CatBoostRegressor(n_estimators=100)
}


results = pd.DataFrame({'Model': [], 'Explained_variance': [], 'MSE': [], 'MAB': [], "R2-score": [], 'Time': []})

for model_name, model in classifiers.items():
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time

    y_pred = model.predict(x_val)

    results = results.append({"Model": model_name,
                              "Explained_variance": metrics.explained_variance_score(y_val, y_pred),
                              "MSE": metrics.mean_squared_error(y_val, y_pred),
                              "MAB": metrics.mean_absolute_error(y_val, y_pred),
                              "R2-score": metrics.r2_score(y_val, y_pred),
                              "Time": total_time},
                             ignore_index=True)

results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
y_pred = classifiers[results_ord['Model'][0]].predict(x_val)

plt.figure()
plt.plot(np.linspace(1, y_val.shape[0],y_val.shape[0]), y_val, label='real', linewidth=1 )
plt.plot(np.linspace(1, y_val.shape[0],y_val.shape[0]), y_pred, linestyle='dashed', label='prediction',linewidth=0.5 )
plt.legend()
plt.show()

print(results)