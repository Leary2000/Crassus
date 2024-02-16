import pandas as pd
import ta
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from ta.trend import MACD
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import save_model

def load_and_process_data(csv_file_path):
  

    df = pd.read_csv(csv_file_path)
    df.set_index('DateTime', inplace=True)  

    df.fillna(method='ffill', inplace=True)   # Forward filling
   
    return df

file_path = 'merged_engineered_candles_data.csv'

df_features = load_and_process_data(file_path) 

# Check the results
print(df_features.isnull().sum())  # Verify handling of missing values
print(df_features.describe())       # Explore the data

usable_columns = ['OKX Open', 'OKX High', 'OKX Low', 'OKX Close', 'OKX Volume', 
                  'Binance Open', 'Binance High', 'Binance Low', 'Binance Close', 'Binance Volume', 'Close Price Difference']

X = df_features[usable_columns].copy()

# Feature engineering
X['OKX Close_lag1'] = X['OKX Close'].shift(1)
X['Binance Volume_lag2'] = X['Binance Volume'].shift(2)
X['OKX Price-Volume Trend'] = X['OKX Close'] * X['OKX Volume']
X['Binance Volatility'] = X['Binance High'] - X['Binance Low']
X['OKX Close_SMA20'] = ta.trend.sma_indicator(X['OKX Close'], window=20)

y = df_features['OKX Percentage Change'].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preparation: Reshaping for CNN input
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 5
X_train, y_train = create_dataset(X_train, y_train, time_steps)
X_test, y_test = create_dataset(X_test, y_test, time_steps)

# Build the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam())

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')


# Predict on the test set
y_pred = model.predict(X_test)

# Calculate additional evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print out the metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')


# Save the model
model_path = 'UNI/FYP/Crassus/your_model.h5'
save_model(model, model_path)
print(f"Model saved to {model_path}")

