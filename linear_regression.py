import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate
from Preprocessing import *

def linear_prediction(df):
    """
    Train a Linear Regression model to predict stock prices and visualize results.
    """
    # Define features and target
    features = ['Open', 'High', 'Low', 'Volume', 'moving_average_10', 'RSI', 'MACD', 'MACD_Signal', '20_SMA', 'BB_Upper', 'BB_Lower']
    X = df[features].fillna(0)
    #X=df.drop('Close',axis=1)
    X=X.fillna(0)
    y = df['Close']

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train Linear Regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predictions
    y_pred = regressor.predict(X_test)
    
    # Model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print Metrics
    metrics_data = [
        ['Mean Absolute Error', mae],
        ['Mean Squared Error', mse],
        ['Root Mean Squared Error', rmse],
        ['R^2 Score', r2]
    ]
    print("Metrics:")
    print(tabulate(metrics_data, headers=['Metric', 'Value'], tablefmt='psql'))



    # Create DataFrame for actual vs. predicted values
    compare = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}, index=y_test.index)
    compare.sort_index(inplace=True)


    # Visualization - Actual vs. Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.plot(compare.index, compare['Actual'], label='Actual', marker='o')
    plt.plot(compare.index, compare['Predicted'], label='Predicted', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs. Predicted Close Prices-Linear Regression')
    plt.legend()
    plt.show()

    # Regression Plot
    sns.regplot(x=y_pred.flatten(), y=y_test.values.flatten(), scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Actual vs. Predicted Price-Linear Regression')
    plt.show()

    return regressor, scaler, mae, mse, rmse, r2, compare



def linear_forecasting(df, scaler, future_days=30):
    # Define features and target
    X = df[['Open', 'High', 'Low', 'Volume',
       'moving_average_10', 'RSI', 'MACD', 'MACD_Signal', '20_SMA', 'BB_Upper',
       'BB_Lower']]
    y = df['Close']
    X=X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit the linear regression model
    regressor = LinearRegression()
    regressor.fit(X, y)

    # Predicting for the future dates
    # Tarih indeksini datetime formatına çevir (eğer değilse)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    #Creating Future Dates
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=future_days, freq='B')

    
    future_features = df.iloc[-1*future_days:][['Open', 'High', 'Low', 'Volume',
       'moving_average_10', 'RSI', 'MACD', 'MACD_Signal', '20_SMA', 'BB_Upper',
       'BB_Lower']]
    #Denormalized future predictions
    future_features_scaled = scaler.transform(future_features)
    future_predictions = regressor.predict(future_features_scaled)

    # Creating a DataFrame for future predictions
    future_prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted': future_predictions
    })
    future_prediction_df.set_index('Date', inplace=True)

    # Create figure with plotly
    fig = go.Figure()

    # Historical data trace
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines+markers', name='Historical Close'))

    # Future predictions trace
    fig.add_trace(go.Scatter(x=future_prediction_df.index, y=future_prediction_df['Predicted'], mode='lines+markers', name='Predicted Close'))

    # Update layout for better interactive controls
    fig.update_layout(
        title='Historical vs Predicted Close Prices',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        yaxis=dict(
            title="Close Price",
            autorange=True,
            type="linear"
        )
    )

    fig.show()

    return regressor, future_prediction_df

