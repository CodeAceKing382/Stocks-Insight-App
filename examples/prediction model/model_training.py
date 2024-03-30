# Importing libraries
import yfinance as yf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

ticker = 'ADANIENT.NS'
temp = yf.download(ticker, period='2y', interval='1d')  # 1-year daily data
temp.dropna(how="any", inplace=True)
ohlcv_data = temp

# SMA Indicator
def SMA(DF, n):
    """Function to calculate Simple Moving Average."""
    df = DF.copy()
    df['SMA'] = df['Adj Close'].rolling(window=n).mean()
    return df['SMA']

# MACD Indicator
def MACD(DF, a=12, b=26, c=9):
    """Function to calculate MACD; typical values: a (fast MA) = 12, b (slow MA) = 26, c (signal line MA) = 9."""
    df = DF.copy()
    df["ma_fast"] = df["Adj Close"].ewm(span=a, min_periods=a).mean()
    # here , decay in terms of a span, which is the number of observations used for calculating the exponentially weighted moving average
    df["ma_slow"] = df["Adj Close"].ewm(span=b, min_periods=b).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["signal"] = df["macd"].ewm(span=c, min_periods=c).mean()

    return df[["macd", "signal"]]

# RSI Indicator
def RSI(DF, n=14):
    """Function to calculate RSI."""
    df = DF.copy()
    df["change"] = df["Adj Close"] - df["Adj Close"].shift(1)  # df["Adj Close"].shift(1) gives previous day's close
    df["gain"] = np.where(df["change"] >= 0, df["change"], 0)  # np.where acts like if condition
    df["loss"] = np.where(df["change"] < 0, -df["change"], 0)
    df["avgGain"] = df["gain"].rolling(window=n).mean()  # Using simple moving average for gain
    df["avgLoss"] = df["loss"].rolling(window=n).mean()  # Using simple moving average for loss
    df["RS"] = df["avgGain"] / df["avgLoss"]  # higher RS value indicates that the average gains are larger than the average losses, suggesting upward momentum
    df["RSI"] = 100 - (100 / (1 + df["RS"]))  # RSI aims to normalize the RS within a range of 0 to 100 (if RS is high , RSI >50 , if RS  is low , RSI<50)
    return df["RSI"]

# ATR Indicator
def ATR(DF, n=14):
    """Function to calculate Average of True Range over certain period """
    df = DF.copy()
    df["H-L"] = df["High"] - df["Low"]  # The range of the current period (high minus low)
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))  # The range from the current high to the previous close.
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))  # The range from the current low to the previous close.
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)  # axis=1 for finding max across row
    df["ATR"] = df["TR"].rolling(window=n).mean()  # Using simple moving average for ATR
    return df["ATR"]

# ADX Indicator
def ADX(DF, n=14):
    """Function to calculate ADX."""
    df = DF.copy()
    # Compare current high with previous high to get positive directional movement and similarly with low for negative movement.
    df["+DM"] = np.where((df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]),df["High"] - df["High"].shift(1), 0)
    df["-DM"] = np.where((df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)),df["Low"].shift(1) - df["Low"], 0)
    # Normalize +DM,-DM by ATR and scale to 100 to get positive , negative directional indicator.
    df["+DI"] = 100 * (df["+DM"].rolling(window=n).mean() / df["ATR"])
    df["-DI"] = 100 * (df["-DM"].rolling(window=n).mean() / df["ATR"])
    df["DX"] = (abs(df["+DI"] - df["-DI"]) / abs(df["+DI"] + df["-DI"])) * 100  # Measure the directional difference between +DI and -DI as a percentage.
    df["ADX"] = df["DX"].rolling(window=n).mean()  # Using simple moving average for ADX
    return df[["ADX", "+DI", "-DI"]]

# Bollinger Bands Indicator
def Boll_Band(DF, n=20):
    """Function to calculate Bollinger Bands."""
    df = DF.copy()
    df["MB"] = df["Adj Close"].rolling(n).mean()  # 20-period SMA
    df["UB"] = df["MB"] + 2 * df["Adj Close"].rolling(n).std(ddof=0)  # upper band taken 2 std.deviations above middle band
    # ddof =0,the standard deviation is calculated by dividing by N, where N is the number of observations, assumes the data set includes the entire population.
    df["LB"] = df["MB"] - 2 * df["Adj Close"].rolling(n).std(ddof=0)  # lower band taken 2 std.deviations below middle band
    df["BB_Width"] = df["UB"] - df["LB"]
    return df[["MB", "UB", "LB", "BB_Width"]]

# OBV Indicator
def OBV(DF):
    """Function to calculate On-Balance Volume."""
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()  # Calculate the daily return as the percentage change of the close price
    df['direction'] = np.where(df['daily_ret'] >= 0, 1,-1)  # positive return , direcction is 1 and 0 for negative return
    df['direction'][0] = 0  # Initialize the first value of 'direction' to 0 to handle the first row (due to pct_change)
    df['vol_adj'] = df['Volume'] * df['direction']  # Adjust volume by direction to account for buying (positive) or selling (negative)
    df['OBV'] = df['vol_adj'].cumsum()  # Cumulatively sum the adjusted volume to get the On-Balance Volume
    return df['OBV']

# Heikin-Ashi Indicator
def Heikin_Ashi(DF):
    df = DF.copy()

    # Heikin Ashi close is the average of open, high, low, close
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    # Heikin Ashi open is the midpoint of the previous HA candle (except for the first one, which is just the open price)
    df['HA_Open'] = 0.0
    for i in range(len(df)):
        if i == 0:
            df['HA_Open'].iloc[i] = df['Open'].iloc[i]
        else:
            df['HA_Open'].iloc[i] = (df['HA_Open'].iloc[i - 1] + df['HA_Close'].iloc[i - 1]) / 2

    # Heikin Ashi high is the maximum of the high, HA_Open, and HA_Close
    df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)

    # Heikin Ashi low is the minimum of the low, HA_Open, and HA_Close
    df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

    # Determine the Heikin Ashi trading signal
    df['HA_Decision'] = 0  # Default to 'Hold'
    # 'Buy' if current HA candle is bullish and the previous was bearish
    df.loc[(df['HA_Close'] > df['HA_Open']) & (df['HA_Close'].shift(1) < df['HA_Open'].shift(1)), 'HA_Decision'] = 1
    # 'Sell' if current HA candle is bearish and the previous was bullish
    df.loc[(df['HA_Close'] < df['HA_Open']) & (df['HA_Close'].shift(1) > df['HA_Open'].shift(1)), 'HA_Decision'] = -1

    return df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'HA_Decision']]

# consolidating each indicator and creating a single column for each indicator result
# 50-day SMA.captures medium-term market trends
ohlcv_data['SMA_50'] = SMA(ohlcv_data, 50)  # 50-day SMA

# MACD: Consolidate into MACD_diff,indicating potential bullish , bearish trends and  reversals.
macd_data = MACD(ohlcv_data)
ohlcv_data['MACD_diff'] = macd_data['macd'] - macd_data['signal']

# ATR: Average True Range, indicating market volatility. Higher ATR values suggest increased volatility, while lower values suggest consolidation
ohlcv_data['ATR'] = ATR(ohlcv_data)

# Bollinger Bands: Bollinger Band Width, a measure of market volatility. wider bands indicate higher volatility, and narrower bands suggest lower volatility.
bollinger_data = Boll_Band(ohlcv_data)
ohlcv_data['BB_Width'] = bollinger_data['BB_Width']

# On-Balance Volume to track volume changes and predict price movements based on volume flow.
ohlcv_data['OBV'] = OBV(ohlcv_data)

# RSI: Relative Strength Index, indicating overbought or oversold conditions.
ohlcv_data['RSI'] = RSI(ohlcv_data)

# ADX: Average Directional Index, measuring the strength of a trend.
ohlcv_data[['ADX', "+DI", "-DI"]] = ADX(ohlcv_data)

# Heikin-Ashi : smooths price data and removes noise to help identify and stay with the trend by averaging price movements.
ohlcv_data[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'HA_Decision']] = Heikin_Ashi(ohlcv_data)

def Momentum_Shift_Indicator(df):
    # Identify RSI crossovers (crossing above 30 for buy signal, below 70 for potential sell signal)
    rsi_buy_signal = (df['RSI'].shift(1) < 30) & (df['RSI'] > 30)
    rsi_sell_signal = (df['RSI'].shift(1) > 70) & (df['RSI'] < 70)

    # Identify MACD crossovers (MACD crossing above signal line for buy, below for sell)
    macd_buy_signal = (df['MACD_diff'].shift(1) < 0) & (df['MACD_diff'] > 0)
    macd_sell_signal = (df['MACD_diff'].shift(1) > 0) & (df['MACD_diff'] < 0)

    # Combine signals: Buy when both RSI and MACD indicate buy, Sell when both indicate sell , else hold
    df['MSI'] = np.where(rsi_buy_signal & macd_buy_signal, 1, np.where(rsi_sell_signal & macd_sell_signal, -1, 0))
    return df['MSI']

def Volatility_Contraction_Indicator(DF):
    df = DF.copy()

    # Calculate the short-term average and long-term average of both ATR and BB_Width
    df['ATR_SMA_short'] = df['ATR'].rolling(window=10).mean()
    df['ATR_SMA_long'] = df['ATR'].rolling(window=30).mean()
    df['BB_Width_SMA_short'] = df['BB_Width'].rolling(window=10).mean()
    df['BB_Width_SMA_long'] = df['BB_Width'].rolling(window=30).mean()

    # Identify contraction in volatility: short-term averages falling below long-term averages
    atr_contraction = df['ATR_SMA_short'] < df['ATR_SMA_long']  # indicates that recent price movements are smaller and less volatile compared to the past
    bb_width_contraction = df['BB_Width_SMA_short'] < df['BB_Width_SMA_long']  # signifies that the price is moving less dramatically, indicating a squeeze or contraction in volatility.

    # Combine signals: Mark periods of volatility contraction
    df['VCI'] = np.where(atr_contraction & bb_width_contraction, 1, 0)
    # VCI typically doesn't have a 'Sell' condition, it's more about spotting potential breakouts
    return df['VCI']

def Heikin_Volume_Indicator(DF):
    df = DF.copy()

    # Initialize a new column for the HVI signal
    df['HVI_Signal'] = 0

    # Define conditions for buy and sell signals

    heikin_ash_buy = df['HA_Close'] > df['HA_Open']  # Heikin Ashi shows bullish trend
    obv_buy = df['OBV'] > df['OBV'].shift(1)  # OBV is increasing

    heikin_ash_sell = df['HA_Close'] < df['HA_Open']  # Heikin Ashi shows bearish trend
    obv_sell = df['OBV'] < df['OBV'].shift(1)  # OBV is decreasing

    # Assign signals based on conditions
    df['HVI_Signal'] = np.where(heikin_ash_buy & obv_buy, 1,np.where(heikin_ash_sell & obv_sell, -1, 0))  # Assign signals based on conditions

    # Return only the HVI signal column
    return df['HVI_Signal']

# ADX: Average Directional Index, measuring the strength of a trend.
ohlcv_data['MSI'] = Momentum_Shift_Indicator(ohlcv_data)

# ADX: Average Directional Index, measuring the strength of a trend.
ohlcv_data['VCI'] = Volatility_Contraction_Indicator(ohlcv_data)

# ADX: Average Directional Index, measuring the strength of a trend.
ohlcv_data['HVI_Signal'] = Heikin_Volume_Indicator(ohlcv_data)

ohlcv_data.dropna(how="any", inplace=True)

# Evaluates trading signals  on the provided indicators and rules and gives decision 'Buy' or 'Sell' or 'Hold'
def evaluate_trading_signals(DF):
    df = DF.copy()

    # Initialize the Decision column with 'Hold' (0)
    df['Decision'] = 0

    # SMA (50-day) Rules
    df['SMA_signal'] = 'Hold'
    df.loc[df['Close'] > df['SMA_50'], 'SMA_signal'] = 'Buy'
    df.loc[df['Close'] < df['SMA_50'], 'SMA_signal'] = 'Sell'

    # MACD_diff Rules
    df['MACD_signal'] = 'Hold'
    df.loc[df['MACD_diff'] > 0, 'MACD_signal'] = 'Buy'
    df.loc[df['MACD_diff'] < 0, 'MACD_signal'] = 'Sell'

    # ATR Rules
    df['ATR_signal'] = 'Hold'
    df.loc[df['ATR'] < df['ATR'].rolling( window=14).mean(), 'ATR_signal'] = 'Buy'  # Low ATR compared to recent history suggests potential breakout from consolidation
    df.loc[df['ATR'] > df['ATR'].rolling(window=14).mean(), 'ATR_signal'] = 'Sell'  # High ATR indicates high volatility, possibly leading to reversals

    # BB_Width Rules
    df['BB_signal'] = 'Hold'
    df.loc[df['BB_Width'] < df['BB_Width'].rolling(window=14).mean(), 'BB_signal'] = 'Buy'  # Narrow BB_Width suggests market consolidation, often preceding a breakout
    df.loc[df['BB_Width'] > df['BB_Width'].rolling(window=14).mean(), 'BB_signal'] = 'Sell'  # Wide BB_Width indicates high volatility, which might occur during tops or significant price movements

    # OBV Rules
    df['OBV_signal'] = 'Hold'
    df.loc[df['OBV'] > df['OBV'].shift(1), 'OBV_signal'] = 'Buy'  # Set to 'Buy' if OBV is increasing, indicating accumulation
    df.loc[df['OBV'] < df['OBV'].shift(1), 'OBV_signal'] = 'Sell'  # Set to 'Sell' if OBV is decreasing, indicating distribution

    # RSI Rules
    df['RSI_signal'] = 'Hold'
    df.loc[df['RSI'] > 30, 'RSI_signal'] = 'Buy'  # Set to 'Buy' if RSI is above 30, possibly exiting oversold conditions
    df.loc[df['RSI'] < 70, 'RSI_signal'] = 'Sell'  # Set to 'Sell' if RSI is below 70, not entering overbought conditions

    # ADX Rules
    df['ADX_signal'] = 'Hold'
    df.loc[(df['ADX'] > 25) & (df['+DI'] > df['-DI']), 'ADX_signal'] = 'Buy'  # Set to 'Buy' if ADX is above 25 and +DI is above -DI, indicating a strong bullish trend
    df.loc[(df['ADX'] > 25) & (df['+DI'] < df['-DI']), 'ADX_signal'] = 'Sell'  # Set to 'Sell' if ADX is above 25 and +DI is below -DI, indicating a strong bearish trend

    # HA_signal
    df['HA_signal'] = 'Hold'
    df.loc[df['HA_Decision'] == 1, 'HA_signal'] = 'Buy'
    df.loc[df['HA_Decision'] == -1, 'HA_signal'] = 'Sell'

    # MSI Signal
    df['MSI_signal'] = 'Hold'
    df.loc[df['MSI'] == 1, 'MSI_signal'] = 'Buy'
    df.loc[df['MSI'] == -1, 'MSI_signal'] = 'Sell'

    # VCI Signal
    df['VCI_signal'] = 'Hold'
    df.loc[df['VCI'] == 1, 'VCI_signal'] = 'Buy'
    # VCI typically doesn't have a 'Sell' condition, it is about spotting potential breakouts

    # HVI Signal
    df['HVI_signal'] = 'Hold'
    df.loc[df['HVI_Signal'] == 1, 'HVI_signal'] = 'Buy'
    df.loc[df['HVI_Signal'] == -1, 'HVI_signal'] = 'Sell'

    # Consolidate all indicator signals into a list
    signals = ['SMA_signal', 'MACD_signal', 'ATR_signal', 'BB_signal', 'OBV_signal', 'RSI_signal', 'ADX_signal',
               'HA_signal', 'HVI_signal', 'MSI_signal', 'VCI_signal']

    for index, row in df.iterrows():
        # 'index' holds the index label, and 'row' is a Series of the row's values.
        buy_signals = sum(row[signal] == 'Buy' for signal in signals)  # Count the 'Buy' signals for the current row across all indicators
        sell_signals = sum(row[signal] == 'Sell' for signal in signals)  # Count the 'Sell' signals for the current row across all indicators

        if buy_signals > sell_signals:
            df.at[index, 'Decision'] = 1  # Sets the value of 'Decision' at the current 'index' to 1 (Buy).
        elif sell_signals > buy_signals:
            df.at[index, 'Decision'] = -1  # Sets the value of 'Decision' at the current 'index' to -1 (Sell).
        # Hold (0) is the default value

    return df['Decision']

# Apply the evaluation function to each DataFrame ohlcv_data

ohlcv_data["decision"] = evaluate_trading_signals(ohlcv_data)

df = ohlcv_data
# Removing additional columns other than the main technical indicators
df.drop(["+DI", "-DI", "HA_Open", "HA_High", "HA_Low", "HA_Close"], axis=1, inplace=True)

# Selecting feature columns (all your indicators)
feature_cols = ['SMA_50', 'MACD_diff', 'ATR', 'BB_Width', 'OBV', 'RSI', 'ADX', 'HA_Decision', 'MSI', 'VCI',
                'HVI_Signal']
X = df[feature_cols]
# Target column
y = df['decision']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize the scaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Fit on the training data and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data with the same scaler
X_test_scaled = scaler.transform(X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # Sigmoid function to map any value to the (0,1) interval .

def compute_cost(X, y, weights, bias):
    m = X.shape[0]  # Number of samples (number of rows in database)
    z = np.dot(X, weights) + bias  # Linear combination of weights, inputs, and bias
    probabilities = sigmoid(z)  # Applying sigmoid function to get probabilities
    # Computing the cost function for logistic regression
    cost = -(1 / m) * np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
    return cost

def gradient_descent(X, y, weights, bias, learning_rate, iterations):
    m = X.shape[0]  # Number of training examples
    cost_history = []  # To store the cost at each iteration, useful for checking convergence

    for i in range(iterations):  # Loop over the number of iterations
        z = np.dot(X, weights) + bias  # Compute the net input (linear combination of weights and inputs plus bias)

        probabilities = sigmoid(z)  # Apply the sigmoid function to compute the probability of each class

        difference = probabilities - y  # Calculate the difference between the predicted probabilities and actual labels

        dW = (1 / m) * np.dot(X.T, difference)  # Calculate the gradient of the cost function w.r.t. weights
        dB = (1 / m) * np.sum(difference)  # Calculate the gradient of the cost function w.r.t. bias

        weights -= learning_rate * dW  # fine-tuning weights by moving against the gradient by the learning rate
        bias -= learning_rate * dB  # fine-tuning bias by moving against the gradient by the learning rate

        cost = compute_cost(X, y, weights,bias)  # Recalculate the cost with updated weights and bias to check for convergence
        cost_history.append(cost)  # Append the new cost to the cost history

    return weights, bias, cost_history  # Return the final updated weights, bias, and the history of cost values

def train_ovr(X, y, class_label, learning_rate=0.01, iterations=1000):
    y_bin = (y == class_label).astype(int)  # Convert multiclass(-1,0,1) to binary classification (1 for the current class, 0 for all others)
    weights = np.zeros(X.shape[1])  # Initialize weights as zeros ( weights is a array in size of number of features )
    bias = 0  # Initialize bias as zero

    weights, bias, _ = gradient_descent(X, y_bin, weights, bias, learning_rate,iterations)  # Perform gradient descent
    return weights, bias

classifiers = []
for class_label in [-1, 0, 1]:
    weights, bias = train_ovr(X_train_scaled, y_train, class_label)  # Train a classifier for each class
    classifiers.append((weights, bias))

def predict_ovr_with_thresholds(X, classifiers, thresholds):
    probability_predictions = []
    # Iterate over each classifier to get probabilities
    for weights, bias in classifiers:
        z = np.dot(X, weights) + bias  # Linear combination
        probabilities = sigmoid(z)  # Apply sigmoid to get probabilities
        probability_predictions.append(probabilities)

    # Convert list of probability predictions to a NumPy array
    predictions = np.array(probability_predictions).T

    # Initialize an array to hold the final class predictions
    final_predictions = np.zeros(predictions.shape[0])

    # Apply thresholds to determine class labels
    for i, class_probs in enumerate(predictions):
        class_label = np.argmax(class_probs) - 1  # Get the class with the highest probability
        if class_probs[class_label + 1] >= thresholds[class_label]:
            final_predictions[i] = class_label  # Assign class label if above threshold
        else:
            final_predictions[i] = 0  # Default to 'hold' if below threshold

    return final_predictions

# Define your thresholds for each class
thresholds = {
    -1: 0.5,  # Threshold for 'sell' (-1)
    0: 1,  # Threshold for 'hold' (0)
    1: 0.5  # Threshold for 'buy' (1)
}

# Apply the updated prediction function with thresholds
predictions_with_thresholds = predict_ovr_with_thresholds(X_test_scaled, classifiers, thresholds)











