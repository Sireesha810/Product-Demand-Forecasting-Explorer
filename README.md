# Product Demand Forecasting Data Explorer

## 📌 Project Overview
This project is a Python-based Data Explorer designed to analyze historical product sales data. The primary goal is to identify seasonal demand patterns, calculate moving averages for trend analysis, and prepare a clean, feature-engineered dataset ready for future machine learning forecasting models (like ARIMA, Prophet, or XGBoost).

## ✨ Features
* **Data Preprocessing:** Handles datetime conversions, sets a time-series index, and fills missing values.
* **Moving Averages:** Calculates 7-day (weekly) and 30-day (monthly) moving averages to smooth out daily noise and highlight trends.
* **Seasonality Analysis:** Extracts month and day-of-week features to identify when demand is highest.
* **Visualizations:** Generates clear charts for historical trends, monthly/weekly seasonality, and time-series decomposition (Trend, Seasonality, Residual).
* **Data Export:** Outputs a clean, organized CSV file (`.csv`) ready to be fed into a forecasting model.

## 🛠️ Technologies Used
* **Python**
* **Pandas** (Data manipulation and time-series indexing)
* **NumPy** (Mathematical operations and synthetic data generation)
* **Matplotlib & Seaborn** (Data visualization)
* **Statsmodels** (Time-series decomposition)

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Sireesha810/Product-Demand-Forecasting-Explorer.git