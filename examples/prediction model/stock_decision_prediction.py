from model_training import *
import yfinance as yf
import pandas as pd

def calculate_indicators(DF):
    # Calculate all the necessary indicators for the DataFrame.

    DF['SMA_50'] = SMA(DF, 50)  # 50-day SMA
    macd_data = MACD(DF)
    DF['MACD_diff'] = macd_data['macd'] - macd_data['signal']
    DF['ATR'] = ATR(DF)
    bollinger_data = Boll_Band(DF)
    DF['BB_Width'] = bollinger_data['BB_Width']
    DF['OBV'] = OBV(DF)
    DF['RSI'] = RSI(DF)
    DF[['ADX', "+DI", "-DI"]] = ADX(DF)
    DF[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'HA_Decision']] = Heikin_Ashi(DF)
    DF['MSI'] = Momentum_Shift_Indicator(DF)
    DF['VCI'] = Volatility_Contraction_Indicator(DF)
    DF['HVI_Signal'] = Heikin_Volume_Indicator(DF)
    DF.dropna(how="any", inplace=True)
    DF.drop(["+DI", "-DI", "HA_Open", "HA_High", "HA_Low", "HA_Close"], axis=1, inplace=True)

    return DF

def predict_on_data(data, classifiers, scaler, feature_cols):
    # Apply the model to make predictions based on the indicators of the given OHLCV data.

    # Calculate indicators for the entire DataFrame
    data_with_indicators = calculate_indicators(data)

    # Select and scale the features
    X = data_with_indicators[feature_cols]
    X_scaled = scaler.transform(X)
    # Use the trained classifiers to make predictions
    predictions = predict_ovr_with_thresholds(X_scaled, classifiers, thresholds)

    # Return the predictions
    return predictions



# List of Nifty 50 tickers
nifty_50_tickers = [
    "ITC.NS", "BAJAJ-AUTO.NS", "BRITANNIA.NS", "LTIM.NS", "BHARTIARTL.NS", "TECHM.NS",
    "HINDALCO.NS", "RELIANCE.NS","TATACONSUM.NS", "KOTAKBANK.NS", "MARUTI.NS", "TCS.NS",
    "COALINDIA.NS", "HDFCLIFE.NS", "TITAN.NS", "NTPC.NS", "INDUSINDBK.NS", "ULTRACEMCO.NS",     "LT.NS", "WIPRO.NS", "APOLLOHOSP.NS", "TATASTEEL.NS", "NESTLEIND.NS", "CIPLA.NS",
   "ADANIENT.NS", "ONGC.NS", "BAJFINANCE.NS", "HEROMOTOCO.NS", "BAJAJFINSV.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "JSWSTEEL.NS", "SBIN.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "AXISBANK.NS", "POWERGRID.NS",
    "DRREDDY.NS", "ASIANPAINT.NS", "HINDUNILVR.NS", "DIVISLAB.NS", "BAJAJFINSV.NS", "M&M.NS",
    "BPCL.NS", "SBILIFE.NS", "UPL.NS"]

# Initialize a DataFrame to hold all predictions
all_predictions = pd.DataFrame()

for ticker in nifty_50_tickers:
    # Download stock data
    stock_data = yf.download(ticker, period='5mo', interval='1d')  # assuming you want daily predictions
    stock_data.dropna(how="any", inplace=True)


    # Generate predictions
    stock_data['prediction'] = predict_on_data(stock_data, classifiers, scaler, feature_cols)

    # Add a column for the ticker symbol
    stock_data['ticker'] = ticker

    # Reset the index to turn the Date index into a column
    stock_data.reset_index(inplace=True)

    # Append the predictions to the all_predictions DataFrame
    all_predictions = pd.concat([all_predictions, stock_data.tail(10)])

import os

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Now you can use file_path to save your DataFrame to CSV
all_predictions.to_csv("./examples/csv/nifty_50_predictions.csv", index=False)

print('Dataset generated and saved to nifty_50_predictions.csv')


import json

# Load the CSV file
df = pd.read_csv("./examples/csv/nifty_50_predictions.csv")

# Define the JSONL file path
jsonl_file_path2= "./examples/data/nifty_50_predictions.jsonl"

# Open the file in write mode
with open(jsonl_file_path2, 'w') as jsonl_file:
    # Iterate over DataFrame rows as (index, Series) pairs
    for _, row in df.iterrows():
        # Convert the row to dictionary, then to string, format into a 'doc' structure
        doc_str = ', '.join(f"{key}: {value}" for key, value in row.to_dict().items())
        jsonl_file.write(json.dumps({"doc": doc_str}) + '\n')


jsonl_file_path1="./examples/data/_stock_variables_info.jsonl"

jsonl_combined_file_path ="./examples/data/stock_predict_total.jsonl"

# Read the contents of both jsonl files
with open(jsonl_file_path1, 'r') as file1, open(jsonl_file_path2, 'r') as file2:
    data1 = file1.readlines()
    data2 = file2.readlines()

# Combine the contents
combined_data = data1 + data2

# Write the combined data to a new jsonl file
with open(jsonl_combined_file_path, 'w') as outfile:
    for entry in combined_data:
        outfile.write(entry)

print(f"Combined JSONL file created at {jsonl_combined_file_path}")

