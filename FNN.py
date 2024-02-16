import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LTSM
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import TimeSeriesSplit




# Load the data
df = pd.read_csv('merged_engineered_candles_data.csv')
print(df.columns)


# Select features and target
features = ['OKX Open', 'OKX High', 'OKX Low', 'OKX Close', 'OKX Volume', 'Binance Open', 'Binance High', 'Binance Low', 'Binance Close', 'Binance Volume']
target = 'OKX Percentage Change'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Callbacks
lr_callback = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Build LSTM model function
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Adjust input shape for LSTM layer
input_shape = (X_train_scaled.shape[1], 1)  # LSTM expects 3D input shape [samples, timesteps, features]
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model = build_lstm_model(input_shape)

# Model summary
model.summary()



history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32)


test_loss = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {test_loss}')

predictions = model.predict(X_test_scaled)
# Compare predictions with y_test
