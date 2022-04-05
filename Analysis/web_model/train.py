import time

import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier, plot_tree
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import data_handler as dh
import settings as st

data = dh.create_data()
x_train, y_train, x_test, y_test = dh.preprocess(data)
tree_classifiers = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, bootstrap=True),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "Skl GBM": GradientBoostingClassifier(n_estimators=100),
        "Skl HistGBM": HistGradientBoostingClassifier(max_iter=100),
        "XGBoost": XGBClassifier(n_estimators=100),
        "LightGBM": LGBMClassifier(n_estimators=100),
        # "CatBoost": CatBoostClassifier(n_estimators=100),
    }

results = pd.DataFrame({'Model': [], 'Precision': [], 'Time': []})

for model_name, model in tree_classifiers.items():
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time

    pred = model.predict(x_test)

    results = results.append({"Model": model_name,
                              "Precision": accuracy_score(y_test, pred) * 100,
                              "Time": total_time},
                             ignore_index=True)

results.reset_index(drop=True, inplace=True)
results_ord = results.sort_values(by=['Precision'], ascending=False, ignore_index=True)

best_pred = tree_classifiers[results_ord['Model'][0]].predict(x_test)
print(confusion_matrix(y_test, best_pred))

# save the best model to disk
filename = f'{results_ord["Model"][0]}.sav'
joblib.dump(tree_classifiers[results_ord['Model'][0]], filename)
