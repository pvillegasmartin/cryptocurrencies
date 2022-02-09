import preprocess_data.volatility.data_prep_volatility as data_preprocess
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, html, dcc
import datetime
import numpy as np


def main():

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
    """
    plt.scatter(df_train_15m[:, 0], df_train_15m[:, 1], c=output_train)
    plt.xlabel('NATR')
    plt.ylabel('total_change_norm')
    plt.show()
    """
    df_15m = data_preprocess.read_data('BTCUSDT-15m.csv')
    df_15m['labels'] = np.resize(output_train,len(df_15m))
    df_15m['datetime'] = df_15m['time'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0))
    df_15m, df_train_15m = df_15m.iloc[:50, :], df_train_15m[:50, 1]
    fig = go.Figure(data=[go.Candlestick(x=df_15m['datetime'],
                                         open=df_15m['Open'],
                                         high=df_15m['High'],
                                         low=df_15m['Low'],
                                         close=df_15m['Close'])])
    """
    fig.add_trace(go.Scatter(
            x=df_15m['datetime'],
            y=df_train_15m,
            showlegend=False,
            line_color='blue'
        ))
    """
    for row in df_15m.index:
        fig.add_vrect(x0=df_15m.iloc[row,:]['datetime']-datetime.timedelta(minutes=15),
                      x1=df_15m.iloc[row,:]['datetime'],
                      fillcolor='red' if df_15m.iloc[row,:]['labels'] == 0 else 'green',
                      opacity=0.1,
                      line_width=0)
    fig.show(renderer="svg")

"""
app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Hello William'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
"""

if __name__ == '__main__':
    main()
    # app.run_server(debug=True)