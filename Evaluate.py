import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the saved model
model = load_model('lstm_model.h5')

# Assuming X_test and y_test are already defined and preprocessed
# If predictions haven't been made:
predictions = model.predict(X_test)

# Convert predictions and y_test to the same shape if necessary
predictions = predictions.flatten()  # Adjust based on your model's output shape

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y_test, predictions)
print(f"R-squared (R²): {r2}")

# Handle division by zero in MAPE calculation
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100 if np.any(y_test) else float('inf')
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
