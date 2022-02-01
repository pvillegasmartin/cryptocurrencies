import preprocess_data.data_preprocessing as data_func
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

if __name__ == '__main__':
    df = data_func.read_data()
    df = data_func.create_indicators(df)
    # TODO calculate output
    df['output'] = 0

    # Divide train/test
    df_train = df.iloc[:int(0.75*len(df)),:]
    df_test = df.drop(df_train.index)

    # Divide input/output
    x_train, x_test = df_train.drop('output', axis=1), df_test.drop('output', axis=1)
    y_train, y_test = df_train.output, df_test.output

    # Clean cache
    del df, df_train, df_test

    # Train model
    model = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
    model.fit(x_train, y_train)
    print("Accuracy of the model: ", round(model.score(x_train, y_train) * 100, 2))

    # Test model
    pred = model.predict(x_test)
    print("Accuracy of testing: ", round(metrics.accuracy_score(y_test, pred) * 100,2))