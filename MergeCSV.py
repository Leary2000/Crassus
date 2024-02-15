import csv
from datetime import datetime

def calculate_percentage_change(current, previous):
    if previous == 0:
        return 0
    try:
        return (current - previous) / previous * 100.0
    except ZeroDivisionError:
        return 0

def merge_and_engineer_features(file1, file2, output_file):
    # Load the first file into a dictionary with DateTime as key
    data1 = {}
    with open(file1, mode='r') as f1:
        reader = csv.DictReader(f1)
        previous_close = 0
        for row in reader:
            row['OKX Spread'] = float(row['OKX High']) - float(row['OKX Low'])
            row['OKX Percentage Change'] = calculate_percentage_change(float(row['OKX Close']), previous_close)
            previous_close = float(row['OKX Close'])
            data1[row['DateTime']] = row
    
    # Load the second file, merge with the first, and engineer features
    merged_data = {}
    with open(file2, mode='r') as f2:
        reader = csv.DictReader(f2)
        previous_close = 0
        for row in reader:
            if row['DateTime'] in data1:
                row['Binance Spread'] = float(row['Binance High']) - float(row['Binance Low'])
                row['Binance Percentage Change'] = calculate_percentage_change(float(row['Binance Close']), previous_close)
                previous_close = float(row['Binance Close'])
                
                # Combine data from both files for matching DateTime
                combined_row = {**data1[row['DateTime']], **row}
                
                # Calculate the Close price difference between OKX and Binance
                combined_row['Close Price Difference'] = float(combined_row['OKX Close']) - float(combined_row['Binance Close'])
                
                merged_data[row['DateTime']] = combined_row
    
    # Write merged data with engineered features to output file
    if merged_data:
        fieldnames = ['DateTime', 'OKX Open', 'OKX High', 'OKX Low', 'OKX Close', 'OKX Volume', 'OKX Spread', 'OKX Percentage Change', 'Binance Open', 'Binance High', 'Binance Low', 'Binance Close', 'Binance Volume', 'Binance Close Time', 'Binance Quote Asset Volume', 'Binance Number of Trades', 'Binance Taker buy base asset volume', 'Binance Taker buy quote asset volume', 'Binance Spread', 'Binance Percentage Change', 'Close Price Difference']
        with open(output_file, mode='w', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged_data.values():
                # Adjust the row dictionary to match the new fieldnames
                adjusted_row = {k.replace(' ', ' '): v for k, v in row.items()}  # Replace spaces with underscores in keys
                writer.writerow(adjusted_row)
        print(f"Merged data with engineered features written to {output_file}")
    else:
        print("No matching DateTime found between files.")

merge_and_engineer_features('okx_candles_data.csv', 'TRBBinance4.csv', 'merged_engineered_candles_data.csv')
