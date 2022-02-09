import os
import time
from math import floor
from zipfile import ZipFile, ZIP_DEFLATED
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

import settings
from lib import *

BinanceClient = Client("", "")


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
    start_ts = int(start_str) if type(start_str) in (int, float) else date_to_milliseconds(start_str)

    # Se le resta 1 minuto para que no pida la vela en construccion.
    end_ts = date_to_milliseconds(end_str) - 60000 if end_str else None

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
                                                   limit=settings.candles_request,
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
                n_requests = floor((end_ts - request[0][0]) / (timeframe * settings.candles_request))
                bar = SlowBar(f'Processing {symbol}', max=n_requests)

            if symbol_existed:
                bar.next()
                output_data += ['\n'.join(
                    [','.join(['binance', symbol] + [str(value) for value in candle[:-1]]) for candle in request])]

                # update our start timestamp using the last value in the array and add the interval timeframe
                start_ts = request[len(request) - 1][0] + timeframe
            else:
                # it wasn't listed yet, increment our start date
                start_ts += timeframe * settings.candles_request

            # to avoid reaching the max requests per minute, it sleeps for the rest of the minute if applicable.
            if idx % settings.requests_minute == 0:
                delta = time.time() - start_timer
                if delta < 59:
                    time.sleep(delta)
                # it also uses this break to write part of the fule
                file.write('\n'.join(output_data)+'\n')
                start_timer = time.time()
                output_data = []

            # check if we received less than the required limit and exit the loop
            if len(request) < settings.candles_request:
                # exit the while loop
                break

        file.write('\n'.join(output_data)+'\n')
        if symbol_existed:
            bar.finish()


if __name__ == '__main__':

    for i, symbol in enumerate(settings.symbols):
        print(f"\n{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} - Starting generation for {symbol}.")
        get_historical_klines(
            start_str=settings.first_date,
            end_str=settings.last_date,
            interval=settings.interval,
            symbol=symbol)
        print(f"\n{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} - Ended generation for {symbol}.")

        # Every X files, it will compress them into a zip file.
        if i % settings.zip_pack == 0:
            date_time_str = datetime.now().strftime('%Y%m%d%H%M%S')
            # give the symbol name to zip if it's the only one
            zip_name = f'{symbol}_{date_time_str}.zip' if settings.zip_pack == 1 else f"hist_pack_{date_time_str}_{i}.zip"
            with ZipFile(zip_name, 'w', ZIP_DEFLATED) as zip:
                # Writing each file one by one
                files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('-hist.csv')]
                for file2zip in files:
                    zip.write(file2zip)
                    # Removes the original file
                    os.remove(file2zip)
