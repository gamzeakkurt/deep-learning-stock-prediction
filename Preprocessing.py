# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Function to normalize data using MinMax Scaling for all features
def min_max_scaling(data):
    """Applies Min-Max scaling to normalize all features (Open, High, Low, Close, Adj Close, Volume)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Normalize all features
    scaled_data = scaler.fit_transform(data)  
    print("Data normalization using Min-Max Scaling completed.")
    return scaler, scaled_data



def denormalize_predictions(y_pred, df):
    """
    Denormalize the predicted values using the MinMaxScaler fitted on the original data.
    
    Args:
        y_pred (np.array): The normalized predicted stock values.
        df (pd.DataFrame): The original dataframe containing the stock prices.

    Returns:
        np.array: The denormalized predicted values.
    """
    # Extract 'Close' column from the dataframe, which was used for normalization
    close_prices = df['Close'].values.reshape(-1, 1)  # Reshape to (n_samples, 1)

    # Create a MinMaxScaler and fit it using the 'Close' prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_prices)  # Fit the scaler on the 'Close' price column

    # Apply the inverse_transform to denormalize the predicted values
    y_pred_denormalized = scaler.inverse_transform(y_pred.reshape(-1, 1))  # Denormalize predictions

    return y_pred_denormalized.flatten()  # Flatten to return a 1D array

def split_and_reshape_data(dataframe, pred_days, company):
    """
    Splits the dataset into training and testing sets, then reshapes it for LSTM models.

    Parameters:
        dataframe (pandas DataFrame): Scaled dataset.
        pred_days (int): Number of previous days used for prediction.
        company (str): Company name.

    Returns:
        X_train, y_train, X_test, y_test: Reshaped datasets for model training and testing.
    """
    prediction_days = pred_days
    
    train_size = int(np.ceil(len(dataframe) * 0.70))  # 70% for training data
    test_size = len(dataframe) - train_size  # Remaining 30% for testing data
    print(f'The training size for {company} is {train_size} rows')
    print(f'The testing size for {company.title()} is {test_size} rows')

    # Use .iloc[] for proper slicing of pandas DataFrame
    train_data = dataframe.iloc[0: train_size, :]  # Use iloc for slicing DataFrame
    test_data = dataframe.iloc[train_size - prediction_days:, :]  # Use iloc for slicing DataFrame

    X_train, y_train, X_test, y_test = [], [], [], []

    # Loop to create X_train and y_train for training data
    for i in range(prediction_days, len(train_data)):
        X_train.append(train_data.iloc[i - prediction_days: i, :].values)  # Features: previous 'pred_days' values for all columns
        y_train.append(train_data.iloc[i, 3])  # Target: next day's 'Close' value (index 3 corresponds to 'Close')

    # Loop to create X_test and y_test for testing data
    for i in range(prediction_days, len(test_data)):
        X_test.append(test_data.iloc[i - prediction_days: i, :].values)  # Features: previous 'pred_days' values for all columns
        y_test.append(test_data.iloc[i, 3])  # Target: next day's 'Close' value (index 3 corresponds to 'Close')

    # Convert the lists to numpy arrays
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    # Reshape the data to be suitable for LSTM model (3D array: samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))  # Number of features (columns) will be dynamic
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))  # Same for test data

    print(f'Data for {company.title()} split successfully')

    return X_train, y_train, X_test, y_test
