
import plotly.graph_objects as go
import datetime
from dash import Dash, html, dcc
import pandas as pd

def plots():
    # read csv
    df_15m = pd.read_csv('C:/Users/Pablo/Desktop/STRIVE/cryptocurrencies/get_data/BTCUSDT-15m.csv', header=0, sep=',',
                         quotechar='"')

    # TODO which columns?
    # select only interesting columns
    df_15m = df_15m.loc[:, ['time', 'Open', 'Close', 'High', 'Low']]

    df_15m['datetime'] = df_15m['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    df_15m = df_15m.iloc[:100,:]
    fig = go.Figure(data=[go.Candlestick(x=df_15m['datetime'],
                                         open=df_15m['Open'],
                                         high=df_15m['High'],
                                         low=df_15m['Low'],
                                         close=df_15m['Close'])])

    return fig

fig = plots()


app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Hello William'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)