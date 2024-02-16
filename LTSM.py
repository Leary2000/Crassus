import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


print(tf.config.list_physical_devices())  # You should see 'CPU' and 'GPU' listed

# Specifically list GPUs
print(tf.config.list_physical_devices('GPU'))
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Temporarily hides CUDA devices



df = pd.read_csv('merged_engineered_candles_data.csv')

# Handle missing values, datetime conversion, etc.
df.set_index('DateTime', inplace=True) 


# Inspect missing data patterns
print(df.isnull().sum())  # Counts missing values per column

# Define columns to use for features
usable_columns = ['OKX Open', 'OKX High', 'OKX Low', 'OKX Close', 'OKX Volume', 
                  'Binance Open', 'Binance High', 'Binance Low', 'Binance Close', 'Binance Volume', 'Close Price Difference']

df_features = df[usable_columns]

# Impute missing values
df_features = df_features.fillna(df_features.mean())



import ta

# Technical Indicators (adjust parameters as needed)
df_features['RSI_14'] = ta.momentum.rsi(df_features['OKX Close'], window=14) 
df_features['SMA_10'] = ta.trend.sma_indicator(df_features['OKX Close'], window=10)
# Add more technical indicators from the 'ta' library

# Lag Features
df_features['OKX Close_lag1'] = df_features['OKX Close'].shift(1)
df_features['Binance Volume_lag2'] = df_features['Binance Volume'].shift(2)
# Add more lag features with different shifts

# Additional Features (Examples)
df_features['OKX Price-Volume Trend'] = df_features['OKX Close'] * df['OKX Volume']
df_features['Binance Volatility'] = df_features['Binance High'] - df_features['Binance Low'] 

#print(df_features)



#                           MODEL                       #


# Target Calculation: Calculate Percentage Change
df_features['OKX Percentage Change'] = df_features['OKX Close'].pct_change() * 100  # Calculates percentage change and scales by 100

df_features.fillna(method='ffill', inplace=True)  # Forward fill
df_features.dropna(inplace=True)  # Drop any remaining NaNs

print(df_features.isnull().sum())
print(df_features.describe())

# Splitting Data
features = df.drop('OKX Percentage Change', axis=1)
target = df['OKX Percentage Change']
# Ensure no infinite values are present
df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
df_features.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Data Preparation for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 5
X_train, y_train = create_dataset(X_train, y_train, time_steps)
X_test, y_test = create_dataset(X_test, y_test, time_steps)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = np.array([scaler.fit_transform(x) for x in X_train])
X_test_scaled = np.array([scaler.transform(x) for x in X_test])

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_lstm_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))

# Model Training
lr_callback = LearningRateScheduler(lambda epoch, lr: lr * 0.9 if epoch > 10 else lr)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[lr_callback, early_stopping])

# Model Evaluation
test_loss = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss}')

predictions = model.predict(X_test_scaled).flatten()

# Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print(f"MAE: {mae}, RMSE: {rmse}, R^2: {r2}, MAPE: {mape}%")

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y_test, predictions)
print(f"R-squared (R²): {r2}")

# Handle division by zero in MAPE calculation
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100 if np.any(y_test) else float('inf')
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")



