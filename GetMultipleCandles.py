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
                'OKX Open': item[1],
                'OKX High': item[2],
                'OKX Low': item[3],
                'OKX Close': item[4],
                'OKX Volume': item[5]
            })

        if data:
            num_candles = len(data)
            # Assuming each candle represents 1 minute, adjust if your interval is different
            minutes_to_advance = num_candles * 60 * 1000  # milliseconds
            current_start_date += minutes_to_advance
            print(current_start_date)

    # Sort the candles by DateTime before saving
    all_candles.sort(key=lambda x: x['DateTime'])

    # Save to CSV
    fieldnames = ['DateTime', 'OKX Open', 'OKX High', 'OKX Low', 'OKX Close', 'OKX Volume']
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
        window_end_timestamp = start_timestamp + 100000000  # Adjust based on your needs
        if window_end_timestamp > end_timestamp:
            window_end_timestamp = end_timestamp
            
        candles = get_Binance_candles(symbol, interval, start_timestamp, window_end_timestamp, 1000)
        time.sleep(1)  # Respect API rate limits
        
        all_candles.extend(candles)
        start_timestamp = window_end_timestamp
        
        reqs += 1
        print(f"Requests: {reqs}, Total Candles: {len(all_candles)}")
    
    filename = "TRBBinance4.csv"
    
    # Define new fieldnames with "Binance" prefix
    fieldnames = ['DateTime', 'Binance Open', 'Binance High', 'Binance Low', 'Binance Close', 'Binance Volume', 'Binance Close Time', 'Binance Quote Asset Volume', 'Binance Number of Trades', 'Binance Taker buy base asset volume', 'Binance Taker buy quote asset volume']
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for candle in all_candles:
            # Prefix each key in the candle dictionary
            prefixed_candle = {'Binance ' + key if key != 'DateTime' else key: value for key, value in candle.items()}
            writer.writerow(prefixed_candle)

    print(f"Data saved to {filename}")

   

    

if __name__ == "__main__":
    EndDate = int(datetime.strptime("13/02/2024", "%d/%m/%Y").timestamp()) * 1000
    StartDate = int(datetime.strptime("10/02/2024", "%d/%m/%Y").timestamp()) * 1000

    multiple_Binance_requests("TRBUSDT", '1m',StartDate, EndDate)
    #multiple_okx_requests("TRB-USDT",'1m',"10/02/2024","13/02/2024")

