#Import libraries

from data_acquisition import *
from FeatureEngineering import *
from Preprocessing import *
from  linear_regression import *
from eda import perform_eda
from lstm_model import *
from result_visualization_lstm import *


# Define tickers
tickers = ['AAPL', 'AMZN', 'TSLA', 'MSFT']
companies = ['Apple', 'Amazon', 'Tesla', 'Microsoft']

#Load data
df = load_stock_data(tickers)

#Perform Exploratory Data Analysis (EDA)
perform_eda(df, tickers, companies)

# Feature Engineering for Apple Stock
df_apple = download_and_process_data('AAPL')


# Preprocessing(Normalizing)
scaler, normalized_data = min_max_scaling(df_apple)
df_normalized = pd.DataFrame(normalized_data, columns=df_apple.columns)
df_normalized["Date"] = df_apple.index
df_normalized=df_normalized.set_index('Date')

#Linear Regression Model
regressor, scaler, mae, mse, rmse, r2, compare_df = linear_prediction(df_normalized)


#Future Forecasting with Linear Regression
regressor, future_predictions = linear_forecasting(df_apple,scaler, future_days=30)

#Saving Future Prediction Results
future_predictions.to_csv('Results/future_predictions-linear-regression.csv')

#Filling Nan Values
df_normalized=df_normalized.fillna(0)

#Reshaping dataset before LSTM 
X_train, y_train, X_test, y_test=split_and_reshape_data(df_normalized, 30, 'Apple')

#Run LSTM Model
model, predictor, y_pred = train_lstm_model(X_train, y_train, X_test, y_test, 'Apple')

#Visualize Model Loss
model_loss(predictor)

#Denormalized Predictions 
y_pred_denormalized=denormalize_predictions(y_pred,df_apple)
y_test_denormalized=denormalize_predictions(y_test,df_apple)

#Visualize Model Prediction
plot_stock_predictions(y_pred_denormalized, y_test_denormalized,df_apple)

#Performance Metrics
prediction_metrics(y_pred_denormalized,y_test_denormalized)

# Reset Index 
df_apple=df_apple.reset_index()


#Forecasting Next 30 Days
forecast_result=forecast_lstm(model,df_apple,df_normalized, future_days=30, n_steps=50)
forecast_result.to_csv('Results/future_predictions_lstm.csv')
if isinstance(forecast_result, tuple):  
    forecast_df = forecast_result[0]  # Extract the first element if it's a tuple
else:
    forecast_df = forecast_result  # If it's already a DataFrame, use it directly


#Ploting Forecasting Result
plot_forecast_results(df_apple, forecast_df)

