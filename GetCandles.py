import requests
import csv
from datetime import datetime, timedelta


def get_okx_candles(symbol, interval, start_date):
    BASE_URL = 'https://www.okx.com/api/v5/market/history-candles'
    params = {
        'instId': symbol,
        'bar': interval,
        'after': start_date,
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json().get('data', [])


    return [
        {
            'DateTime': datetime.utcfromtimestamp(int(item[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'Open': item[1],
            'High': item[2],
            'Low': item[3],
            'Close': item[4],
            'Volume': item[5]
        }
        for item in data
    ]


def get_Binance_candles(symbol, interval, start_timestamp, end_timestamp, limit):
    
    BASE_URL = "https://api.binance.com/api/v3/klines"

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit' : limit,
        'startTime' : start_timestamp,
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    candle_data = [
        {
            # Revert back to DateTime
            'DateTime': item[0],
            'Open': item[1],
            'High': item[2],
            'Low': item[3],
            'Close': item[4],
            'Volume': item[5],
        }
        for item in data
    ]
    
    return candle_data

def get_coingecko_candles(symbol, from_timestamp, to_timestamp):

    BASE_URL = "https://api.coingecko.com/api/v3"
    
    endpoint = f"/coins/{symbol}/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': from_timestamp,
        'to': to_timestamp,
        'resolution': 'minute'
    }

    response = requests.get(BASE_URL + endpoint, params=params)
    data = response.json()

    # The returned data structure contains prices, market_caps, and total_volumes arrays.
    # Each array contains pairs [timestamp, value]
    prices = data.get('prices', [])

    candles = []
    for price in prices:
        timestamp, value = price
        datetime_str = datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        candles.append({'DateTime': datetime_str, 'prices': value})

    return candles


    candle_data = [
        {
            # Revert back to DateTime
            'DateTime': item[0],
            'Open': item[1],
            'High': item[2],
            'Low': item[3],
            'Close': item[4],
            'Volume': item[5],
        }
        for item in data
    ]
    
    return candle_data

def get_Kucoin_candles(symbol,start,end,interval):
    BASE_URL = " https://api.kucoin.com/api/v1/market/candles"
    params = {
        "symbol" : symbol,
        "type" : interval
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    data = data.get('data', [])

    for candle in data:
        print(candle)

    return 1

def save_to_csv(data, filename="candles_data2.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)