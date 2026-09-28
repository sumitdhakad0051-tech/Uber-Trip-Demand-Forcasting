"""Baseline Uber trip demand model (Random Forest) on a single month of 2014 data.

Adapted from the project brief. Two fixes vs. the original sample:
  * The raw file has no 'Trips' column, so trips are aggregated per hour first.
  * seaborn countplot now uses x=... (required in recent seaborn versions).
"""
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %% Load the dataset
data = pd.read_csv("uber-raw-data-apr14.csv")
print(data.info())

# %% Data preprocessing
data["Date/Time"] = pd.to_datetime(data["Date/Time"])
data["Hour"] = data["Date/Time"].dt.hour
data["Day"] = data["Date/Time"].dt.day
data["DayOfWeek"] = data["Date/Time"].dt.dayofweek
data["Month"] = data["Date/Time"].dt.month

# %% Exploratory data analysis
plt.figure(figsize=(10, 6))
sns.countplot(x=data["Hour"])
plt.title("Trips per Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Trips")
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x=data["DayOfWeek"])
plt.title("Trips per Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Trips")
plt.show()

# %% Feature engineering: aggregate to hourly trip counts (target = 'Trips')
hourly = (
    data.groupby(["Month", "Day", "DayOfWeek", "Hour"])
    .size()
    .reset_index(name="Trips")
)

X = hourly[["Hour", "Day", "DayOfWeek", "Month"]]
y = hourly["Trips"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %% Model building
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

# %% Model evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# %% Visualization of predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Trips")
plt.ylabel("Predicted Trips")
plt.title("Actual vs Predicted Trips")
plt.show()
