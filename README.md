# 📊 Yahoo Finance Stock Market Analysis (AAPL, MSFT, AMZN, TSLA)  

## 🚀 Introduction  

Stock market prediction is a crucial area in financial analysis. Prices of stocks are influenced by various factors, such as market trends, economic indicators, and investor sentiment. This project focuses on analyzing and forecasting stock prices of **Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), and Tesla (TSLA)** using deep learning.  

Using **Yahoo Finance data**, we apply **Exploratory Data Analysis (EDA), Feature Engineering, Preprocessing, and Long Short-Term Memory (LSTM) Regression modeling** to predict stock prices.  

---

## 🎯 Objectives  

✅ Retrieve stock market data using `yfinance` 📈  
✅ Perform **EDA** to visualize trends & correlations 📊  
✅ Extracting features like **RSI, MACD,Bollinger Bands, Moving Averages vs.**  
✅ Preprocess the data for deep learning (normalization, handling missing values)  
✅ Implement an **LSTM and Linear Regression models** for stock price forecasting 🧠  
✅ Evaluate predictions using **RMSE, MAE, and R² scores**  
✅ Compare actual vs. predicted stock prices 📉  

---

## 🏗️ Project Workflow  

🔹 **Step 1:** Data Collection (Yahoo Finance API)  
🔹 **Step 2:** Exploratory Data Analysis (EDA)  
🔹 **Step 3:** Feature Engineering (Technical Indicators)  
🔹 **Step 4:** Data Preprocessing (Normalization, Reshaping)  
🔹 **Step 5:** LSTM & Linear Regression Models Training & Prediction  
🔹 **Step 6:** Model Evaluation (Error Metrics)  
🔹 **Step 7:** Results & Visualization  

---

## ⚙️ Installation & Setup  

### 📌 **1. Clone the Repository**  
```bash
git clone https://github.com/gamzeakkurt/deep-learning-stock-prediction.git
cd deep-learning-stock-prediction


```

### 📌 **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 📌 **3. Run the Jupyter Notebook**  
```bash
jupyter notebook
```
Open `Stock_Market_Analysis.ipynb` and execute the cells.

---

## 📦 Dependencies  

The project requires the following libraries:  

```txt
yfinance
quantstats
ta
PyPortfolioOpt
pandas==1.3.5  
numpy  
matplotlib  
seaborn  
scikit-learn  
tensorflow  
keras
plotly  
```

You can install them using:  
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow keras plotly
```

---

## 📊 Exploratory Data Analysis (EDA)  

✅ **Stock Price Trends:** Visualize historical stock prices over time  
✅ **Moving Averages & Indicators:** Compute SMA, EMA, RSI, and MACD  
✅ **Correlation Analysis:** Analyze relationships between different stocks  

🔍 **Example Visualization:**  

<p align="center">
  <img src="images/stock_trend.png" width="600">
</p>

---

## 🏗️ LSTM Model Overview  

### **Why LSTM?**  
LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) that are effective for **time-series forecasting** due to their ability to capture long-term dependencies.  

**Model Architecture:**  
- Input Shape: `(n_timesteps, 1)`  
- LSTM Layers: 2 stacked LSTMs  
- Fully Connected (Dense) Layer  
- Activation: ReLU  
- Optimizer: Adam  
- Loss Function: Mean Squared Error (MSE)  

---

## 📈 Results & Evaluation  

The model is evaluated using the following metrics:  

📌 **Root Mean Square Error (RMSE)**  
📌 **Mean Absolute Error (MAE)**  
📌 **R² Score (Coefficient of Determination)**  

🔍 **Example Prediction Plot:**  

<p align="center"><img width="600" alt="Screenshot 2025-01-30 at 22 13 28" src="https://github.com/user-attachments/assets/2277dac4-5b66-4cda-bb63-9d1929128101" />


</p>

---

## 📌 Future Improvements  

🔹 Include **news sentiment analysis** for better predictions  
🔹 Optimize **hyperparameters** for improved accuracy  
🔹 Compare LSTM results with **ARIMA, XGBoost, and CNN models**  

---

## 📜 License  

This project is licensed under the **MIT License**.  

---


## 📬 Contact  

For any questions or suggestions, feel free to reach out. 



