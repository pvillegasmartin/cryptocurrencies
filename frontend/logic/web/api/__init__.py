from flask import Flask, render_template, request, redirect
from flask_restful import Api, Resource, reqparse
from flask_sqlalchemy import SQLAlchemy
import requests
import json
from api.get_data import *
import time

app = Flask(__name__, template_folder='../templates', static_folder='../static')
api = Api(app)
app.config.from_object("api.config.Config")
db = SQLAlchemy(app)


class Crypto(db.Model):
    __tabletime__ = 'Crypto'
    id = db.Column(db.String(64), primary_key=True, unique=True)
    coin = db.Column(db.String(64), unique=False, nullable=False)
    time = db.Column(db.String(64), unique=False, nullable=False)
    high = db.Column(db.Float(), unique=False, nullable=False)
    low = db.Column(db.Float(), unique=False, nullable=False)
    open = db.Column(db.Float(), unique=False, nullable=False)
    close = db.Column(db.Float(), unique=False, nullable=False)
    volume = db.Column(db.Float(), unique=False, nullable=False)

    def __init__(self, coin, time, high, low, open, close, volume):
        self.id = coin + str(time)
        self.coin = coin
        self.time = time
        self.high = high
        self.low = low
        self.open = open
        self.close = close
        self.volume = volume

class Crypto_api(Resource):

    def get(self):
        args_parser = reqparse.RequestParser()
        args_parser.add_argument('coin', type=str)

        args = args_parser.parse_args()
        coin = args['coin']

        try:
            crypto_info = db.session().query(Crypto).filter_by(coin=coin).all()
            coin_list = []
            time_list = []
            high_list = []
            low_list = []
            close_list = []
            open_list = []
            volume_list = []
            for timeframe in crypto_info:
                coin_list.append(timeframe.coin)
                time_list.append(timeframe.time)
                high_list.append(timeframe.high)
                low_list.append(timeframe.low)
                close_list.append(timeframe.close)
                open_list.append(timeframe.open)
                volume_list.append(timeframe.volume)
            return {'coin':coin_list, 'time':time_list, 'high':high_list, 'low':low_list, 'open':open_list, 'close':close_list, 'volume':volume_list}

        except:
            return {'ERROR': "Couldn't find coin"}

        return {'coin':coin}

    def post(self):
        args_parser = reqparse.RequestParser()
        args_parser.add_argument('coin', type=str)
        args_parser.add_argument('time', type=str)
        args_parser.add_argument('high', type=float)
        args_parser.add_argument('low', type=float)
        args_parser.add_argument('open', type=float)
        args_parser.add_argument('close', type=float)
        args_parser.add_argument('volume', type=float)

        args = args_parser.parse_args()
        coin = args['coin']
        time = args['time']
        high = args['high']
        low = args['low']
        open = args['open']
        close = args['close']
        volume = args['volume']

        try:
            db.session.add(Crypto(coin=coin, time=time, high=high, low=low, open=open, close=close, volume=volume))
            db.session.commit()
            return {'coin':coin, 'time':time}

        except:
            return {'time': []}

api.add_resource(Crypto_api, '/crypto')


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/<coin>', methods=['GET', 'POST'])
def crypto_detail(coin):
    req = requests.get(f'http://127.0.0.1:5000/crypto?coin={coin}')
    all_info = json.loads(req.content)

    if request.method == 'POST':
        try:
            last_time = max(all_info['time'])
        except:
            last_time = 1640991600000

        try:

            df_new = get_historical_klines(start_str=last_time,end_str='now UTC',interval='1h',symbol=coin)
            for index, row in df_new.iterrows():
                requests.post(f'http://localhost:5000/crypto?coin={row["symbol"]}&time={row["time"]}&high={row["High"]}&low={row["Low"]}&open={row["Open"]}&close={row["Close"]}&volume={row["Volume"]}')

            return redirect(f'http://localhost:5000/{coin}')

        except:
            return render_template('coin.html', data=all_info)

    else:
        return render_template('coin.html', data=all_info)