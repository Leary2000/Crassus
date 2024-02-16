import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

# from Signals import generate_signals

tfd = tfp.distributions

# Load the dataset
df = pd.read_csv('Crassus\TRBBinance.csv')

# Feature Engineering: Adding lag features
for lag in [1, 2, 3, 5, 10]:  # Example lags
    df[f'lag_close_{lag}'] = df['Close'].shift(lag)

# Calculate the target variable as percentage change
df['Target'] = ((df['Close'].shift(-1) - df['Close']) / df['Close']) * 100
df.dropna(inplace=True)

# Selecting features and target variable for the model
feature_cols = ['Close', 'SMA_20', 'EMA_20', 'MACD', 'MACD_signal', 'RSI_EWMA', 'RSI_SMA', 'middle_band', 'upper_band', 'lower_band', 'Number of Trades'] + [f'lag_close_{lag}' for lag in [1, 2, 3, 5, 10]]
X = df[feature_cols]
y = df['Target']

# Scale the features
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)

# Define the time series split
tscv = TimeSeriesSplit(n_splits=5)

# Define the Bayesian Neural Network
def build_bnn_model(input_shape, output_units, hidden_units=50):
    # Define the prior for the weights and biases
    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1), reinterpreted_batch_ndims=1)

    # Define the posterior for the weights and biases
    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=tf.nn.softplus(t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])

    # Create the BNN model
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(input_shape=input_shape,
                                    units=hidden_units,
                                    make_prior_fn=prior,
                                    make_posterior_fn=posterior,
                                    activation='relu'),
        tfp.layers.DenseVariational(units=2,  # Output mean and standard deviation
                                    make_prior_fn=prior,
                                    make_posterior_fn=posterior,
                                    activation=None),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :output_units], scale=tf.math.softplus(t[..., output_units:]))
        )
    ])

    return model

# Build the model with the corrected output layer
bnn_model = build_bnn_model(input_shape=(X_scaled.shape[1],), output_units=1)


# Build the model
bnn_model = build_bnn_model(input_shape=(X_scaled.shape[1],), output_units=1)

# Compile the model
bnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=lambda y, p_y: -p_y.log_prob(y))

# Initialize an empty list to store RMSE scores
rmse_scores = []

# Train the model and evaluate
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the BNN
    bnn_model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Predict and evaluate (using RMSE)
    y_pred_dist = bnn_model(X_test)
    y_pred = y_pred_dist.mean().numpy().flatten()

    print(f"NaNs in predictions: {np.isnan(y_pred).sum()}, NaNs in actual values: {np.isnan(y_test).sum()}")


    mask = ~np.isnan(y_pred) & ~np.isnan(y_test)
    y_pred_clean = y_pred[mask]
    y_test_clean = y_test[mask]

    rmse = sqrt(mean_squared_error(y_test_clean, y_pred_clean))
    rmse_scores.append(rmse)

# Calculate and print the average RMSE
average_rmse = sum(rmse_scores) / len(rmse_scores)
print(f'Average RMSE over all time-series splits: {average_rmse}')

# Save the model
bnn_model.save('bnn_model')
