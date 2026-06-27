# =====================================================
# Stock Price Prediction using Linear Regression
# =====================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# Load Dataset
# =====================================================

data = pd.read_csv("aapl_us_2025.csv")

print("=" * 60)
print("First 5 Rows")
print("=" * 60)
print(data.head())

# =====================================================
# Dataset Information
# =====================================================

print("\nDataset Information")
print("=" * 60)
print(data.info())

print("\nStatistical Summary")
print("=" * 60)
print(data.describe())

# =====================================================
# Check Missing Values
# =====================================================

print("\nMissing Values")
print("=" * 60)
print(data.isnull().sum())

# =====================================================
# Convert Date Column
# =====================================================

data['Date'] = pd.to_datetime(data['Date'])

# =====================================================
# Visualization 1
# Closing Price
# =====================================================

plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], color='blue')
plt.title("Apple Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.grid(True)
plt.show()

# =====================================================
# Visualization 2
# Trading Volume
# =====================================================

plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Volume'], color='green')
plt.title("Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.show()

# =====================================================
# Visualization 3
# Open vs Close
# =====================================================

plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Open'], label="Open")
plt.plot(data['Date'], data['Close'], label="Close")

plt.title("Open vs Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Create Future Prediction Column
# =====================================================

future_days = 30

data['Prediction'] = data['Close'].shift(-future_days)

# =====================================================
# Feature and Target
# =====================================================

X = np.array(data[['Close']])[:-future_days]

y = np.array(data['Prediction'])[:-future_days]

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Train Model
# =====================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# =====================================================
# Prediction
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# Evaluation
# =====================================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("=" * 60)

print("Mean Absolute Error :", mae)

print("Mean Squared Error  :", mse)

print("Root Mean Squared Error :", rmse)

print("R2 Score :", r2)

# =====================================================
# Actual vs Predicted Plot
# =====================================================

plt.figure(figsize=(12,6))

plt.plot(y_test, label="Actual Price")

plt.plot(predictions, label="Predicted Price")

plt.title("Actual vs Predicted Stock Price")

plt.xlabel("Test Data")

plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.show()

# =====================================================
# Predict Future 30 Days
# =====================================================

future = np.array(data[['Close']])[-future_days:]

future_prediction = model.predict(future)

print("\nPredicted Stock Prices for Next 30 Days")
print("=" * 60)

for i, price in enumerate(future_prediction, start=1):
    print(f"Day {i:2d} : ${price:.2f}")

# =====================================================
# Future Prediction Graph
# =====================================================

plt.figure(figsize=(12,6))

plt.plot(
    range(1, future_days + 1),
    future_prediction,
    marker='o'
)

plt.title("Future 30-Day Stock Price Prediction")

plt.xlabel("Future Days")

plt.ylabel("Predicted Price")

plt.grid(True)

plt.show()

# =====================================================
# Compare Last 30 Actual Prices with Predictions
# =====================================================

last_actual = data['Close'].tail(future_days).values

plt.figure(figsize=(12,6))

plt.plot(last_actual, label="Last Actual Prices")

plt.plot(future_prediction, label="Predicted Prices")

plt.title("Actual vs Future Prediction")

plt.xlabel("Days")

plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.show()

# =====================================================
# End
# =====================================================

print("\nProject Completed Successfully!")