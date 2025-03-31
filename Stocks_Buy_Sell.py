import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# Download historical data for Gold (XAU/USD) with hourly interval
data = yf.download('GC=F', start='2023-01-01', end='2024-08-22', interval='1h')

# Prepare the data
data['Date'] = data.index
data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)  # Remove timezone information if present
data['Hours'] = (data['Date'] - data['Date'].min()).dt.total_seconds() / 3600  # Convert dates to the number of hours

# Use only the 'Close' price for prediction
X = data['Hours'].values.reshape(-1, 1)
y = data['Close'].values

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Project out to the end of 2025
end_date = pd.to_datetime('2025-12-31')
projection_period_hours = int((end_date - data['Date'].max()).total_seconds() / 3600)
future_hours = np.arange(data['Hours'].max() + 1, data['Hours'].max() + projection_period_hours + 1).reshape(-1, 1)
future_dates = [data['Date'].max() + timedelta(hours=i) for i in range(1, projection_period_hours + 1)]
future_prices = model.predict(future_hours)

# Combine historical data with projected data
forecast_data = pd.DataFrame({'Date': future_dates, 'Projected_Close': future_prices})
forecast_data['Hours'] = future_hours

# Recompute SMA for both historical and projected data
data['SMA_24'] = data['Close'].rolling(window=24).mean()  # 24-hour SMA
forecast_data['SMA_24'] = forecast_data['Projected_Close'].rolling(window=24).mean()

# Define buy/sell signals for historical data
data['Signal'] = 0
data.loc[data['Close'] > data['SMA_24'], 'Signal'] = 1  # Buy signal
data.loc[data['Close'] < data['SMA_24'], 'Signal'] = -1  # Sell signal

# Apply the same strategy to projected data
forecast_data['Signal'] = 0
forecast_data.loc[forecast_data['Projected_Close'] > forecast_data['SMA_24'], 'Signal'] = 1  # Buy signal
forecast_data.loc[forecast_data['Projected_Close'] < forecast_data['SMA_24'], 'Signal'] = -1  # Sell signal

# Create figure and axis objects
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot historical and projected prices
ax1.plot(data['Date'], data['Close'], label='Historical Gold Prices', color='blue')
ax1.plot(forecast_data['Date'], forecast_data['Projected_Close'], label='Projected Gold Prices', color='red')
ax1.plot(data['Date'], data['SMA_24'], label='24-Hour SMA (Historical)', color='green', linestyle='--')
ax1.plot(forecast_data['Date'], forecast_data['SMA_24'], label='24-Hour SMA (Projected)', color='orange', linestyle='--')

# Buy/Sell signals for historical data
ax1.scatter(data[data['Signal'] == 1]['Date'], data[data['Signal'] == 1]['Close'], marker='^', color='g', label='Buy Signal (Historical)')
ax1.scatter(data[data['Signal'] == -1]['Date'], data[data['Signal'] == -1]['Close'], marker='v', color='r', label='Sell Signal (Historical)')

# Buy/Sell signals for projected data
ax1.scatter(forecast_data[forecast_data['Signal'] == 1]['Date'], forecast_data[forecast_data['Signal'] == 1]['Projected_Close'], marker='^', color='lime', label='Buy Signal (Projected)')
ax1.scatter(forecast_data[forecast_data['Signal'] == -1]['Date'], forecast_data[forecast_data['Signal'] == -1]['Projected_Close'], marker='v', color='darkred', label='Sell Signal (Projected)')

# Mark the point where projection starts
ax1.axvline(x=data['Date'].max(), color='red', linestyle='--', label='Projection Start')

# Set the y-axis limit to a price range from 1750 to 3000
ax1.set_ylim(1750, 3000)

# Add titles and labels
ax1.set_title('Gold Price Projection Until 2025 with Buy/Sell Signals (Hourly Data)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Gold Price (USD)')
ax1.legend()
ax1.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
