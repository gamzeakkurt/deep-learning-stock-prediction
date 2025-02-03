import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

def build_lstm_model():
    """
    Build and return a BiLSTM model for time-series prediction.
    """
    model = Sequential([
        Conv1D(32, 3, strides=1, activation='relu', input_shape=[30, 18]),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer, predicting the 'Close' price
    ])

    model.compile(optimizer=Adam(), loss=Huber(), metrics=['mse', 'mae'])
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, company):
    """
    Train an LSTM model on the given data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        company: Name of the stock/company

    Returns:
        model: Trained LSTM model
        history: Training history
        y_pred: Predictions on test set
    """

    print(f'========= Training LSTM Model for {company} =========')

    # Build model
    model = build_lstm_model()

    # Train mod
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

    # Predict on test data
    y_pred = model.predict(X_test)

    return model, history, y_pred



def prediction_metrics(y_pred,y_test):
    """
    the predictions and actual values, calculates RMSE, MAE, and R-squared metrics.

    Args:
        company (str): Name of the company
        scaler (MinMaxScaler): The MinMaxScaler used to scale the data
        model: The trained LSTM model
        X_test (numpy.ndarray): The testing features (shape: (n_samples, time_steps, num_features))
        y_test (numpy.ndarray): The true values

    Returns:
        y_pred (numpy.ndarray): The predicted values (denormalized)
        y_test (numpy.ndarray): The true values (denormalized)
        """
       # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        # Calculate MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_test, y_pred)

        # Calculate R-squared (R²)
    r2 = r2_score(y_test, y_pred)

        # Print all metrics
    print(f'The RMSE for Apple is {rmse}')
    print(f'The MAE for Apple is {mae}')
    print(f'The R-squared (R²) for Apple is {r2}')

    


def forecast_lstm(model, df, normalized_df, future_days=30, n_steps=50):
    """
    Forecast the next `future_days` using a trained LSTM model.

    Args:
        model: Trained LSTM model.
        df (DataFrame): Original DataFrame with stock prices (should have a DateTime index).
        normalized_df (np.array): Normalized dataset used as LSTM input.
        future_days (int): Number of days to predict (default: 30).
        n_steps (int): Number of time steps for LSTM input (default: 50).

    Returns:
        forecast_df (DataFrame): DataFrame with forecasted dates and denormalized prices.
    """

    print("\n========= Forecasting Next 30 Days =========")

    # Ensure normalized_df is a NumPy array
    if isinstance(normalized_df, pd.DataFrame):
        normalized_df = normalized_df.values  # Convert to array

    # Validate input shape
    if len(normalized_df.shape) != 2 or normalized_df.shape[1] != 18:
        raise ValueError(f"Expected normalized_df shape (X, 18), but got {normalized_df.shape}")

    # Get test data (last 50 days)
    x_input = normalized_df[-n_steps:].reshape((1, n_steps, 18))  # Reshape for LSTM input
    temp_input = x_input[0].tolist()  # Convert to a properly structured list of lists

    lstm_op = []

    # Ensure df index is datetime format
    df.index = pd.to_datetime(df['Date'])
    last_date = df.index[-1]

    # Generate forecast dates (business days only)
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

    # **Fit MinMaxScaler on the original DataFrame**
    scaler = MinMaxScaler()
    df_close = df[['Close']]  # Select the 'Close' price column
    df_scaled = scaler.fit_transform(df_close)  # Fit and transform only the Close price column

    # Generate predictions iteratively
    for i in range(future_days):
        # Convert temp_input back to a NumPy array before reshaping
        x_input = np.array(temp_input[-n_steps:])  # Take last 50 time steps
        x_input = x_input.reshape((1, n_steps, 18))  # Reshape for LSTM input

        # Predict next step (normalized output)
        predicted = model.predict(x_input, verbose=0).flatten()  # Ensure 1D shape
        
        # Adjust shape handling based on model output
        if predicted.shape[0] == 1:
            predicted_features = np.zeros(18)  # Create an empty array with 18 features
            predicted_features[0] = predicted[0]  # Store the predicted Close price
        else:
            predicted_features = predicted 

        # Append only the first feature (Close price) to output
        lstm_op.append(predicted_features[0])

        # Append full predicted feature set & maintain `n_steps` length
        temp_input.append(predicted_features.tolist())  # Append adjusted feature array
        temp_input = temp_input[1:]  # Keep only last `n_steps` elements

    # Convert lstm_op to NumPy array for inverse transformation
    lstm_op = np.array(lstm_op).reshape(-1, 1)  # Reshape to (30, 1) for inverse scaling

    # Prepare for denormalization: Create an array with the correct number of features (11)
    n_features_in = df_scaled.shape[1]  # The number of features the scaler was originally fitted on
    lstm_op_with_features = np.zeros((lstm_op.shape[0], n_features_in))  # (30, 1)
    lstm_op_with_features[:, 0] = lstm_op.flatten()  # Place predicted Close price in the first column

    # Manually denormalize the predicted values (using the min and max of Close)
    X_min = scaler.data_min_[0]  # Minimum value of Close Price
    X_max = scaler.data_max_[0]  # Maximum value of Close Price

    # Manually denormalize the predictions (flattened lstm_op)
    denormalized_predictions = lstm_op.flatten() * (X_max - X_min) + X_min

    # Debugging: Check the first few denormalized values
    #print(f"Manually Denormalized Predictions: {denormalized_predictions[:5]}")

    # Create DataFrame with predicted dates & prices
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': denormalized_predictions})
    
    # Ensure that forecast_df has the same length as forecast_dates
    if len(forecast_df) != len(forecast_dates):
        raise ValueError(f"The length of forecast_df ({len(forecast_df)}) does not match the length of forecast_dates ({len(forecast_dates)}).")

    forecast_df.set_index('Date', inplace=True)

    print("\nFinal Forecasted Prices:")
    print(forecast_df)

    return forecast_df



