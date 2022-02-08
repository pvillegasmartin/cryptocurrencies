import preprocess_data.volatility.data_prep_volatility as data_preprocess
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objects as go

if __name__ == '__main__':
    df_15m = data_preprocess.read_data('BTCUSDT-15m.csv')
    df_15m = data_preprocess.create_indicators(df_15m)
    df_train_15m_scaled, df_test_15m, scaler = data_preprocess.prepare(df_15m)

    # Clean cache
    del df_15m

    # Train model
    # TODO define how many groups
    model = KMeans(n_clusters=2, random_state=0).fit(df_train_15m_scaled)
    output_train = model.labels_

    # Test model
    df_test_15m_scaled = scaler.transform(df_test_15m)
    output_test = model.predict(df_test_15m_scaled)

    # Unscale values to see real data on plots
    df_train_15m = scaler.inverse_transform(df_train_15m_scaled)

    # Graphing the results
    plt.scatter(df_train_15m[:, 0], df_train_15m[:, 1], c=output_train)
    plt.xlabel('NATR')
    plt.ylabel('total_change_norm')
    plt.show()