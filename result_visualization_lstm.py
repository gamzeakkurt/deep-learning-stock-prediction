import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Preprocessing import *

def plot_forecast_results(df, forecast_df):
    """
    Plots actual vs. predicted prices.

    Args:
        df (DataFrame): Original DataFrame with actual stock prices.
        forecast_df (DataFrame): DataFrame with forecasted prices.

    Returns:
        None (Displays the plot)
    """
    # Ensure df index is in datetime format
    df.index = pd.to_datetime(df.index)
    last_date = df.index[-1]  # Get the last available date in the dataset

    # Generate actual dates for the last 30 business days
    dates_actual = pd.date_range(end=last_date - pd.Timedelta(days=1), periods=30, freq='B')

    # Generate forecast dates for the next 30 business days
    forecast_dates = forecast_df.index

    # Plot actual vs. predicted prices
    plt.figure(figsize=(10, 5))

    # Plot the actual data
    plt.plot(dates_actual, df['Close'][-30:].values, label='Actual Price', marker='o', linestyle='-')

    # Plot the predicted data
    plt.plot(forecast_dates, forecast_df['Predicted Price'].values, label='Predicted Price', marker='x', linestyle='--')

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Forecasting for Apple')

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # Show legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()



def model_loss(predictor):
    """
    Plot the training and validation loss of a model.
    
    Args:
        predictor: The model's history object containing 'loss' and 'val_loss'.
    """
    # Create a figure and axes for the plot
    fig, axes = plt.subplots()

    # Set the main title for the figure
    plt.suptitle('Model Loss')

    # Plot the training loss and validation loss
    axes.plot(predictor.epoch, predictor.history['loss'], label='Training Loss')
    axes.plot(predictor.epoch, predictor.history['val_loss'], label='Validation Loss')

    # Set the title and labels for the axes
    axes.set_title('Apple')  # Customize this based on the model you're working with
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Loss')

    # Customize the tick parameters for better readability
    axes.xaxis.set_tick_params()
    axes.yaxis.set_tick_params()

    # Add a legend to the plot
    axes.legend(loc='upper left')

    # Show the plot
    plt.show()



def create_dataframes_for_plots(dataframe, y_pred):
    """
    Split the original dataframe into training and testing dataframes
    and align the predicted values with the test set.
    
    Args:
        dataframe (pd.DataFrame): The original dataframe containing the stock data.
        y_pred (np.array): The predicted stock values for the test set.
    
    Returns:
        plot_train (pd.DataFrame): Training data with original stock prices.
        plot_test (pd.DataFrame): Testing data with both actual and predicted stock prices.
    """
    # Split the data
    training_data_len = int(np.ceil(len(dataframe) * 0.70))
    
    # Create train and test dataframes
    plot_train = dataframe.iloc[:training_data_len].copy()
    plot_test = dataframe.iloc[training_data_len:].copy()

    # Align Predictions with the test set
    plot_test.loc[:, 'Predictions'] = y_pred.flatten()  # Ensure predictions are 1D

    return plot_train, plot_test

    plt.figure(figsize=(16,6))
    plt.title('Stock Price Prediction Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)

    # Plot training and validation data along with predictions
    plt.plot(train.index, train['Close'], label='Train', color='blue')
    plt.plot(valid.index, valid['Close'], label='Validation', color='green')
    plt.plot(valid.index, valid['Predictions'], label='Predictions', linestyle='dashed', color='red')

    # Set legend and display the plot
    plt.legend(loc='lower right')
    plt.show()



        


def plot_stock_predictions(dataframe, y_pred, df):
    """
    Plot the stock prices with training, validation, and predicted prices.
    
    Args:
        dataframe (pd.DataFrame): The original dataframe containing the stock data.
        y_pred (np.array): The predicted stock values for the test set.
        df (pd.DataFrame): The original dataframe used for denormalization.
    """
   

    # Create train and test dataframes
    plot_train, plot_test = create_dataframes_for_plots(df, y_pred)

    # Ensure train and validation sets maintain the original index
    training_data_len = int(np.ceil(len(df) * 0.70))
    train = df.iloc[:training_data_len].copy()
    valid = df.iloc[training_data_len:].copy()

    # Correctly align Predictions in `valid`
    valid['Predictions'] = plot_test['Predictions']

    # 🛠️ Ensure `valid` and `Predictions` align in length
    print(f"Validation set size: {valid.shape}, Predictions size: {plot_test['Predictions'].shape}")

    # Plot the data
    plt.figure(figsize=(16,6))
    plt.title('Stock Price Prediction Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)

    # Fix: Use original index for proper alignment
    plt.plot(train.index, train['Close'], label='Train')


    plt.plot(valid.index, valid['Close'], label='Validation')

    plt.plot(valid.index, valid['Predictions'], label='Predictions', linestyle='dashed')

    # Ensure the x-axis (date) is correctly aligned
    plt.legend(loc='lower right')
    plt.show()
