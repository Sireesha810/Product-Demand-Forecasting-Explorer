import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os

# Ignore minor warning messages for a clean output
warnings.filterwarnings('ignore')

# Set a nice visual style for our charts
sns.set_theme(style="whitegrid")

# ==========================================
# 1. DATA INGESTION (GENERATE FAKE DATA)
# ==========================================
print("1. Loading data...")

# Create 2 years of daily dates
dates = pd.date_range(start='2022-01-01', end='2023-12-31')
np.random.seed(42) # Keeps random numbers consistent

# Simulate sales: A trend + weekly seasonality + random noise
trend = np.linspace(100, 200, len(dates))
weekly_seasonality = np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 20
noise = np.random.normal(0, 15, len(dates))
sales = trend + weekly_seasonality + noise

# Put it into a Pandas DataFrame
df = pd.DataFrame({'Date': dates, 'Sales': sales})
df['Sales'] = df['Sales'].clip(lower=0) # Prevent negative sales

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
print("2. Preprocessing data...")

# Convert 'Date' column to datetime format and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure continuous daily frequency (fill any missing days)
df = df.asfreq('D')

# If there are any missing values, fill them using linear interpolation
df['Sales'] = df['Sales'].interpolate()

# ==========================================
# 3. FEATURE ENGINEERING (MOVING AVERAGES)
# ==========================================
print("3. Calculating moving averages...")

# Calculate 7-day (weekly) and 30-day (monthly) moving averages
df['MA_7'] = df['Sales'].rolling(window=7).mean()
df['MA_30'] = df['Sales'].rolling(window=30).mean()

# ==========================================
# 4. EXTRACT SEASONAL FEATURES
# ==========================================
print("4. Extracting time-based features...")

df['Year'] = df.index.year
df['Month'] = df.index.month
df['DayOfWeek'] = df.index.day_name()
df['DayOfYear'] = df.index.dayofyear

# ==========================================
# 5. VISUALIZATIONS (THE EXPLORER)
# ==========================================
print("5. Generating visualizations (Close charts to continue)...")

fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# Chart 1: Sales & Moving Averages
axes[0].plot(df.index, df['Sales'], label='Daily Sales', alpha=0.4, color='blue')
axes[0].plot(df.index, df['MA_7'], label='7-Day Moving Avg', color='orange', linewidth=2)
axes[0].plot(df.index, df['MA_30'], label='30-Day Moving Avg', color='red', linewidth=2)
axes[0].set_title('Historical Product Sales & Moving Averages')
axes[0].set_ylabel('Sales')
axes[0].legend()

# Chart 2: Monthly Seasonality
monthly_avg = df.groupby('Month')['Sales'].mean().reset_index()
sns.barplot(data=monthly_avg, x='Month', y='Sales', ax=axes[1], palette='viridis')
axes[1].set_title('Seasonal Pattern: Average Sales by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Average Sales')

# Chart 3: Weekly Seasonality
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_avg = df.groupby('DayOfWeek')['Sales'].mean().reindex(day_order).reset_index()
sns.barplot(data=weekly_avg, x='DayOfWeek', y='Sales', ax=axes[2], palette='magma')
axes[2].set_title('Seasonal Pattern: Average Sales by Day of Week')
axes[2].set_xlabel('Day of Week')
axes[2].set_ylabel('Average Sales')

plt.tight_layout()
plt.show()

# Chart 4: Time Series Decomposition (Trend vs Seasonality vs Noise)
decomposition = seasonal_decompose(df['Sales'], model='additive', period=30)
fig2 = decomposition.plot()
fig2.set_size_inches(12, 8)
fig2.suptitle('Time Series Decomposition', fontsize=16)
plt.show()

# ==========================================
# 6. EXPORT PREPARED DATASET
# ==========================================
print("6. Preparing final dataset for forecasting models...")

# Select the columns we want to keep for the future model
final_df = df[['Sales', 'MA_7', 'MA_30', 'Year', 'Month', 'DayOfWeek', 'DayOfYear']]

# Drop the first 30 days because MA_30 will be 'NaN' (empty) for those days
final_df = final_df.dropna()

# Define the exact file name and output directory location
OUTPUT_DIR = r"D:\Dev\Demand_Explorer"
file_name = "processed_demand_data.csv"

# Combine directory and filename
full_path = os.path.join(OUTPUT_DIR, file_name)

# Save the clean data to the CSV file
final_df.to_csv(full_path)

print("=====================================================")
print("Project Complete! File successfully saved.")
print(f"📍 LOCATION: {full_path}")
print("=====================================================")