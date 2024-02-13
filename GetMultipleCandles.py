import requests
import time
from datetime import datetime, timedelta
from GetCandles import get_Binance_candles, get_okx_candles,get_coingecko_candles, save_to_csv,  get_Kucoin_candles


import csv

def multiple_okx_requests(symbol, interval, start_date_str, end_date_str, filename="okx_candles_data.csv"):
    BASE_URL = 'https://www.okx.com/api/v5/market/history-candles'
    start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
    end_date = datetime.strptime(end_date_str, "%d/%m/%Y")
    all_candles = []

    # Convert start_date to milliseconds since epoch
    current_start_date = int(start_date.timestamp()) * 1000
    end_date_ms = int(end_date.timestamp()) * 1000
    print(end_date_ms - current_start_date)

    while current_start_date < end_date_ms:
        params = {
            'instId': symbol,
            'bar': interval,
            'after': current_start_date,
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json().get('data', [])

        if not data:
            print("No more data received.")
            break

        for item in data:
            all_candles.append({
                'DateTime': item[0],
                'Open': item[1],
                'High': item[2],
                'Low': item[3],
                'Close': item[4],
                'Volume': item[5]
            })

        # Assuming data is returned in reverse chronological order
        last_candle_date = data[-1][0]
        current_start_date = int(datetime.strptime(last_candle_date, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()) * 1000 + 1

    # Sort the candles by DateTime before saving
    all_candles.sort(key=lambda x: x['DateTime'])

    # Save to CSV
    fieldnames = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for candle in all_candles:
            writer.writerow(candle)

    print(f"Data saved to {filename}")




def multiple_Binance_requests(symbol, interval, start_timestamp, end_timestamp):
    all_candles = []
    reqs = 0
    
    
    while start_timestamp < end_timestamp:
       # print(start_timestamp, end_timestamp)
        window_end_timestamp = start_timestamp + 100000000  # 30 days in milliseconds
        #print(window_end_timestamp)
        if window_end_timestamp > end_timestamp:
            window_end_timestamp = end_timestamp
            
        
        candles = get_Binance_candles(symbol, interval, start_timestamp, window_end_timestamp, 1000)
        time.sleep(1)

        
        all_candles.extend(candles)
        start_timestamp = window_end_timestamp
        
        reqs += 1
        print(f"Requests: {reqs}, Total Candles: {len(all_candles)}")
        save_to_csv(all_candles, filename = f"TRBBinance4.csv")

       # if reqs % 10 == 0:
        #    print(f"Requests: {reqs}, Total Candles: {len(all_candles)}")
         #   save_to_csv(all_candles, filename = f"TRBBinance2.csv")


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
    EndDate = int(datetime.strptime("10/02/2024", "%d/%m/%Y").timestamp()) * 1000
    StartDate = int(datetime.strptime("21/01/2024", "%d/%m/%Y").timestamp()) * 1000

    #multiple_Binance_requests("TRBUSDT", '1m',StartDate, EndDate)
    multiple_okx_requests("TRB-USDT",'1m',"21/01/2024","10/02/2024")
    #get_Kucoin_candles("LRC-USDT", 1,1,"1min")


