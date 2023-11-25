import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor  # Using a more complex model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from math import sqrt

# Load the dataset
df = pd.read_csv('TRBBinance.csv')

# Feature Engineering: Adding lag features
for lag in [1, 2, 3, 5, 10]:  # Example lags
    df[f'lag_close_{lag}'] = df['Close'].shift(lag)
print(1)

# Calculate the target variable as percentage change
df['Target'] = ((df['Close'].shift(-1) - df['Close']) / df['Close']) * 100
df.dropna(inplace=True)

# Selecting features and target variable for the model
feature_cols = ['Close', 'SMA_20', 'EMA_20', 'MACD', 'MACD_signal', 'RSI_EWMA', 'RSI_SMA', 'middle_band', 'upper_band', 'lower_band', 'Number of Trades'] + [f'lag_close_{lag}' for lag in [1, 2, 3, 5, 10]]
X = df[feature_cols]
y = df['Target']

sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)

# Define the time series split
tscv = TimeSeriesSplit(n_splits=5)
print(2)
# Initialize a more complex model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # Example parameters

print(3)
# Cross-validation
rmse_scores = []
count = 0;
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)
    print(count)
    count = count + 1
print(4)

# Calculate average RMSE over all splits
average_rmse = sum(rmse_scores) / len(rmse_scores)
print(f'Average RMSE over all time-series splits: {average_rmse}')

# Save the scaler and model to files for later use
joblib.dump(sc_X, 'scaler.joblib')
joblib.dump(regressor, 'regressor.joblib')

new_data_df = pd.read_csv('TRBBinance2.csv')

new_data_df['High-Low Spread'] = new_data_df['High'] - new_data_df['Low']
new_data_df['Close-Open Change'] = new_data_df['Close'] - new_data_df['Open']

# Calculate technical indicators
new_data_df['SMA_20'] = new_data_df['Close'].rolling(window=20).mean()
new_data_df['EMA_20'] = new_data_df['Close'].ewm(span=20, adjust=False).mean()
ema12 = new_data_df['Close'].ewm(span=12, adjust=False).mean()
ema26 = new_data_df['Close'].ewm(span=26, adjust=False).mean()
new_data_df['MACD'] = ema12 - ema26
new_data_df['MACD_signal'] = new_data_df['MACD'].ewm(span=9, adjust=False).mean()
delta = new_data_df['Close'].diff()
up, down = delta.clip(lower=0), -delta.clip(upper=0)
roll_up1 = up.ewm(span=14).mean()
roll_down1 = down.ewm(span=14).mean()
RS1 = roll_up1 / roll_down1
new_data_df['RSI_EWMA'] = 100.0 - (100.0 / (1.0 + RS1))
roll_up2 = up.rolling(window=14).mean()
roll_down2 = down.rolling(window=14).mean()
RS2 = roll_up2 / roll_down2
new_data_df['RSI_SMA'] = 100.0 - (100.0 / (1.0 + RS2))

new_data_df['middle_band'] = new_data_df['SMA_20']
new_data_df['upper_band'] = new_data_df['middle_band'] + 2 * new_data_df['Close'].rolling(window=20).std()
new_data_df['lower_band'] = new_data_df['middle_band'] - 2 * new_data_df['Close'].rolling(window=20).std()

new_data_df.dropna(inplace=True)

new_data_df.dropna(inplace=True)

# Load the scaler and model
sc_X = joblib.load('scaler.joblib')
regressor = joblib.load('regressor.joblib')

# Features for prediction
features_new_data = new_data_df[['Close', 'SMA_20', 'EMA_20', 'MACD', 'MACD_signal', 'RSI_EWMA', 'RSI_SMA', 'middle_band', 'upper_band', 'lower_band', 'Number of Trades']]
features_new_data_scaled = sc_X.transform(features_new_data)

# Make predictions
predictions_new_data = regressor.predict(features_new_data_scaled)

# Creating a DataFrame for comparison
predictions_df = pd.DataFrame(predictions_new_data, columns=['Predicted Price Change %'], index=new_data_df.index)
new_data_df['Actual Price Change %'] = ((new_data_df['Close'].shift(-1) - new_data_df['Close']) / new_data_df['Close']) * 100
combined_df = new_data_df.join(predictions_df)

combined_df.dropna(inplace=True)

# Compare predictions with actual percentage changes
actual_change = combined_df['Actual Price Change %']
predicted_change = combined_df['Predicted Price Change %']
comparison = pd.DataFrame({'Predicted Price Change %': predicted_change, 'Actual Price Change %': actual_change})



# Error metrics
mae = mean_absolute_error(actual_change, predicted_change)
mse = mean_squared_error(actual_change, predicted_change)
rmse = sqrt(mse)
mape = np.mean(np.abs((actual_change - predicted_change) / actual_change)) * 100
r2_score = comparison.corr().loc['Predicted Price Change %', 'Actual Price Change %'] ** 2

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'R^2 Score: {r2_score}')
print()
print(comparison.describe())
