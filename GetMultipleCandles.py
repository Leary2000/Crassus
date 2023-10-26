import requests
import time
from datetime import datetime, timedelta
from GetCandles import get_Binance_candles, get_okx_candles,get_coingecko_candles, save_to_csv,  get_Kucoin_candles

import csv

def multiple_okx_requests(symbol, interval, start_date, rolling_window_days=30):
    all_candles = []
    reqs = 0

    while True:
        candles = get_okx_candles(symbol, interval, start_date)
        if not candles:
            print("No data received.")
            break

        all_candles.extend(candles)

        start_date = start_date - 6000000

        reqs += 1
        if(reqs % 10 == 0):
            print(f"Requests made: {reqs}")
            save_to_csv(all_candles, filename=f"candles_data_TRB_USDT.csv")

    if all_candles:
        save_to_csv(all_candles)


def multiple_Binance_requests(symbol, interval, start_timestamp):
    all_candles = []
    reqs = 0
    
    while start_timestamp < end_timestamp:
        window_end_timestamp = start_timestamp + 100000000  # 30 days in milliseconds
        
        if window_end_timestamp > end_timestamp:
            window_end_timestamp = end_timestamp
            
        
        candles = get_Binance_candles(symbol, interval, start_timestamp, window_end_timestamp, 10)
        time.sleep(100000)

        
        if not candles:
            break
        
        all_candles.extend(candles)
        start_timestamp = window_end_timestamp
        
        reqs += 1
        if reqs % 10 == 0:
            print(f"Requests: {reqs}, Total Candles: {len(all_candles)}")
            save_to_csv(all_candles)


def multiple_coingecko_requests(symbol, vs_currency='usd', total_minutes=100000):


    end_timestamp = datetime.utcnow()
    start_timestamp = end_timestamp - timedelta(minutes=100)
    
    all_candles = []
    minutes_per_request = 100  # CoinGecko's limit for 1-minute candles
    reqs = 0
    while total_minutes > 0:
        end_timestamp = start_timestamp
        start_timestamp = end_timestamp - timedelta(minutes=100)


        # Convert datetime objects to UNIX timestamps
        to_timestamp = int(end_timestamp.timestamp())
        from_timestamp = int(start_timestamp.timestamp())

        candles = get_coingecko_candles(symbol, from_timestamp, to_timestamp)
        all_candles.extend(candles)
        total_minutes = 0
        
        #total_minutes -= minutes_per_request
        time.sleep(1)  # To prevent reaching API rate limits
        #reqs += 1
        #if(reqs % 10 == 0):
         #   print(f"Requests made: {reqs}")
          #  print(all_candles)
           # save_to_csv(all_candles, filename=f"candles_data_rubic.csv")
    
    return all_candles


    

if __name__ == "__main__":
    #EndDate = int(datetime.strptime("10/10/2023", "%d/%m/%Y").timestamp()) * 1000
    #multiple_okx_requests("TRB-USDT", '1m', EndDate)

    get_Kucoin_candles("LRC-USDT", 1,1,"1min")


