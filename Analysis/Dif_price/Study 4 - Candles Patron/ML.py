import pandas as pd
import numpy as np
import talib
import time
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from graphviz import Source
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

from sklearn.metrics import confusion_matrix
from sklearn import pipeline
from sklearn import metrics

import pickle


def create_data(file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-4h.csv', period='4H', output=10,
                periods_out=6, kind='+', past_candles=15, comparison='Close', training=True):
    df = pd.read_csv(file)
    df['Datetime'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    col_study = ['Datetime', 'Close', 'High', 'Low', 'Open']
    df = df[col_study]

    # convert the column (it's a string) to datetime type
    datetime_series = pd.to_datetime(df['Datetime'])
    # create datetime index passing the datetime series
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df = df.set_index(datetime_index)
    # we don't need the column anymore
    df.drop('Datetime', axis=1, inplace=True)
    df = df.sort_index()

    for i in range(1, past_candles):
        df['Close-' + str(i)] = (df['Close'].shift(i) - df[comparison]) / df[comparison] * 100
        df['High-' + str(i)] = (df['High'].shift(i) - df[comparison]) / df[comparison] * 100
        df['Low-' + str(i)] = (df['Low'].shift(i) - df[comparison]) / df[comparison] * 100
        df['Open-' + str(i)] = (df['Open'].shift(i) - df[comparison]) / df[comparison] * 100

    # Calculate output
    if kind == '+':
        # POSITIVE
        df['Output_aux'] = df['Close'].shift(-periods_out).rolling(periods_out).max()
        df['Output'] = ((df['Output_aux'] - df['Close']) / df['Close']) > (output / 100)
    else:
        # NEGATIVE
        df['Output_aux'] = df['Close'].shift(-periods_out).rolling(periods_out).min()
        df['Output'] = ((df['Output_aux'] - df['Close']) / df['Close']) < (-output / 100)

    df.drop(['Output_aux', 'Open', 'High', 'Low'], axis=1, inplace=True)

    # Delete first rows where we can't have some indicators values
    df.dropna(inplace=True)

    # To not have any data on the training, is the final test
    if training:
        df = df[df.index.year < 2022]

    # Split train / test
    y = df['Output']
    x = df.drop(['Output'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

    # Scale data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_train_scaled = x_train
    x_test_scaled = x_test

    return x_train_scaled, y_train, x_test_scaled, y_test, x, y, scaler


def ML_train(x_train, y_train, x_test, y_test, profit, kind):
    tree_classifiers = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, bootstrap=True, class_weight='balanced_subsample'),
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
                                  "Precision": metrics.precision_score(y_test, pred) * 100,
                                  "Time": total_time},
                                 ignore_index=True)

    results.reset_index(drop=True, inplace=True)
    results_ord = results.sort_values(by=['Precision'], ascending=False, ignore_index=True)

    best_pred = tree_classifiers[results_ord['Model'][0]].predict(x_test)
    print(confusion_matrix(y_test, best_pred))

    # save the best model to disk
    filename = f'{results_ord["Model"][0]}_profit{kind}{profit}.sav'
    pickle.dump(tree_classifiers[results_ord['Model'][0]], open(filename, 'wb'))

    return tree_classifiers


if __name__ == '__main__':
    # --- VARIABLES TO DEFINE ---
    kind = '-'
    training = False
    profit = 5
    periods_out = 6
    past_candles = 15
    comparison = 'Close'
    if not training:
        filename = f'Extra Trees_profit-5.sav'
    # ----------------------------

    fee = 0.04
    x_train, y_train, x_test, y_test, x_original, y_original, scaler = create_data(
        file='C:/Users/Pablo/Desktop/PMG/00Versions/get_data/BTCUSDT-4h.csv',
        period='4H', output=profit, periods_out=periods_out, past_candles=past_candles, comparison=comparison,
        kind=kind, training=training)
    if training:
        models = ML_train(x_train, y_train, x_test, y_test, profit, kind)
    else:

        # ---- VERIFY MODEL ----
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        # result = loaded_model.score(x_test, y_test)
        x_scale = scaler.transform(x_original)
        x_original['Output'] = y_original
        x_original['Pred'] = loaded_model.predict(x_scale)

        # Return in case after the periods is still opened
        x_original['Final_close'] = (x_original['Close'].shift(-periods_out) - x_original['Close']) / x_original[
            'Close']

        for eval_year in [2019, 2020, 2021, 2022]:
            print(f'------ {eval_year} ------')
            # Check over a period
            x_new = x_original[x_original.index.year == eval_year]
            true_positives = x_new[(x_new['Pred'] == True) & (x_new['Output'] == True)]['Close'].count()
            false_positives = x_new[(x_new['Pred'] == True) & (x_new['Output'] == False)]['Final_close']
            print(
                f'True positives: {true_positives} - {round(true_positives * 100 / (true_positives + false_positives.count()), 2)}%; False positives: {false_positives.count()} - {round(false_positives.count() * 100 / (true_positives + false_positives.count()), 2)}%')
            if kind == '+':
                # POSITIVE
                final_profit = true_positives * profit + false_positives.sum() * 100 - 2 * fee * (
                        true_positives + false_positives.count())  # 2 cause buy and sell
            else:
                # NEGATIVE
                final_profit = true_positives * profit - false_positives.sum() * 100 - 2 * fee * (
                        true_positives + false_positives.count())  # 2 cause buy and sell

            print(round(final_profit, 2))

        if filename.split()[0] == "Decision":
            graph = Source(
                export_graphviz(loaded_model, out_file=None, feature_names=x_new.columns[:-3], class_names=True))
            graph.format = 'png'
            graph.render('dtree_render', view=True)
