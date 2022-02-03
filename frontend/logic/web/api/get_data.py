import os
import time
from math import floor
from zipfile import ZipFile, ZIP_DEFLATED
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
from datetime import datetime
import pytz
import dateparser

BinanceClient = Client("", "")

def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds

    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/

    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds

    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str

    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms

def get_historical_klines(symbol, interval, start_str, end_str=None):
    """Get Historical Klines from Binance
    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    :param symbol: Name of symbol pair e.g BNBBTC
    :param interval: Biannce Kline interval
    :param start_str: Start date string in UTC format
    :param end_str: optional - end date string in UTC format
    :return: list of OHLCV values
    """
    # init our list
    output_data = []

    # convert interval to useful value in seconds
    timeframe = interval_to_milliseconds(interval)

    # convert our date strings to milliseconds
    start_ts = int(start_str)

    # Se le resta 60 minuto para que no pida la vela en construccion.
    end_ts = date_to_milliseconds(end_str) - 60000*60 if end_str else None

    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False

    # start timer and index
    start_timer = time.time()
    idx = 0

    with open(f'{symbol}-hist.csv', 'w') as file:
        # First line contains the header.
        file.write(
            'exchange,symbol,time,Open,High,Low,Close,Volume,CloseTime,'
            'QuoteAssetVolume,NumberOfTrades,TakerBuyBaseAssetVolume,TakerBuyQuoteAssetVolume\n')
        while True:
            # fetch the klines from start_ts up to max entries or the end_ts if set
            idx += 1
            try:
                request = BinanceClient.get_klines(symbol=symbol,
                                                   interval=interval,
                                                   limit=1000,
                                                   startTime=start_ts,
                                                   endTime=end_ts
                                                   )
            except (BinanceAPIException, BinanceRequestException) as e:
                print(e)
                break

            # handle the case where our start date is before the symbol pair listed on Binance
            if not symbol_existed and len(request):
                symbol_existed = True

                # calculate the number of requests and init the status  q
                n_requests = floor((end_ts - request[0][0]) / (timeframe * 1000))


            if symbol_existed:
                output_data += ['\n'.join(
                    [','.join(['binance', symbol] + [str(value) for value in candle[:-1]]) for candle in request])]

                # update our start timestamp using the last value in the array and add the interval timeframe
                start_ts = request[len(request) - 1][0] + timeframe
            else:
                # it wasn't listed yet, increment our start date
                start_ts += timeframe * 1000

            # to avoid reaching the max requests per minute, it sleeps for the rest of the minute if applicable.
            if idx % 1200 == 0:
                delta = time.time() - start_timer
                if delta < 59:
                    time.sleep(delta)
                # it also uses this break to write part of the fule
                file.write('\n'.join(output_data)+'\n')
                start_timer = time.time()
                output_data = []

            # check if we received less than the required limit and exit the loop
            if len(request) < 1000:
                # exit the while loop
                break

        file.write('\n'.join(output_data)+'\n')

    df = pd.read_csv(f'{symbol}-hist.csv', header=0, sep=',')
    df.time = df.time.apply(lambda x: datetime.fromtimestamp(x/1000.0))
    return df