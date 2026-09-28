"""Uber Trips Forecasting with XGBoost, Random Forest, Gradient Boosted Trees + Ensemble.

Cells are separated by '# %%' so this opens as a notebook in VS Code / Jupytext.
Data: Apr-Sep 2014 files (uber-raw-data-*14.csv) from the FiveThirtyEight/TLC Uber dataset.
"""
# %% 1. Imports and helper functions
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

DATA_DIR = "data"  # folder containing uber-raw-data-*14.csv


def plot_decomposition(result):
    plt.figure(figsize=(22, 18))
    for i, (series, label) in enumerate(
        [
            (result.observed, "Observed"),
            (result.trend, "Trend"),
            (result.seasonal, "Seasonality"),
            (result.resid, "Residuals"),
        ],
        start=1,
    ):
        plt.subplot(4, 1, i)
        plt.plot(series, label=label, lw=1)
        plt.legend(loc="upper left")
    plt.show()


def calculate_error(pred, actual):
    """Manual MAPE (note: divides by the prediction, as in the original notebook)."""
    errors = [abs((a - p) / p) for a, p in zip(actual, pred)]
    return sum(errors) / len(errors)


def plot_predictions(plots, title):
    """plots: list of (x, y, label, linestyle, color)."""
    plt.figure(figsize=(18, 8))
    for x, y, label, linestyle, color in plots:
        plt.plot(x, y, label=label, linestyle=linestyle, color=color, lw=1)
    plt.xlabel("Date")
    plt.ylabel("Trips")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.show()


def create_lagged_features(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


# %% 2. Read the 2014 data and resample to hourly counts
# Original notebook used os.walk('/kaggle/input') and dropped the last file;
# a glob on the 2014 monthly files is more robust.
files = sorted(glob.glob(os.path.join(DATA_DIR, "uber-raw-data-*14.csv")))
uber2014 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

uber2014["Date/Time"] = pd.to_datetime(uber2014["Date/Time"], format="%m/%d/%Y %H:%M:%S")
uber2014 = uber2014.sort_values(by="Date/Time")
uber2014 = uber2014.rename(columns={"Date/Time": "Date"})
uber2014.set_index("Date", inplace=True)

hourly_counts = uber2014["Base"].resample("h").count()
uber2014 = hourly_counts.reset_index()
uber2014.columns = ["Date", "Count"]
uber2014.set_index("Date", inplace=True)

print(uber2014.head())

# %% 3. Choose train / test split
print(uber2014.index.min(), uber2014.index.max())

plt.figure(figsize=(20, 8))
plt.plot(uber2014["Count"], linewidth=1, color="darkslateblue")
plt.xticks(rotation=30, ha="right")
plt.show()

result = seasonal_decompose(uber2014["Count"], model="add", period=24 * 1)
plot_decomposition(result)

cutoff_date = "2014-09-15 00:00:00"
plt.figure(figsize=(20, 8))
plt.plot(result.trend, linewidth=1, color="gray")
plt.axvline(x=pd.Timestamp(cutoff_date), color="red", linestyle="--", linewidth=1)
plt.xticks(rotation=30, ha="right")
plt.show()

# Trend rises in September, so a time-based cutoff is used instead of an 80/20 split.
uber2014_train = uber2014.loc[:cutoff_date]
uber2014_test = uber2014.loc[cutoff_date:]

uber2014_test.rename(columns={"Count": "TEST SET"}).join(
    uber2014_train.rename(columns={"Count": "TRAINING SET"}), how="outer"
).plot(figsize=(15, 5), title="Train / Test Sets", style="-", lw=1)
plt.show()

# %% Lagged features (24-hour window)
window_size = 24
X_train, y_train = create_lagged_features(uber2014_train["Count"].values, window_size)

test_data = np.concatenate(
    [uber2014_train["Count"].values[-window_size:], uber2014_test["Count"].values]
)
X_test, y_test = create_lagged_features(test_data, window_size)

seed = 12345
tscv = TimeSeriesSplit(n_splits=5)

# %% 4. XGBoost
xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed)
xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs=-1,
    verbose=1,
)
xgb_grid_search.fit(X_train, y_train)
print("Best XGBoost parameters:", xgb_grid_search.best_params_)

xgb_predictions = xgb_grid_search.best_estimator_.predict(X_test)
plot_predictions(
    [
        (uber2014_test.index, uber2014_test["Count"], "Test", "-", "darkslateblue"),
        (uber2014_test.index, xgb_predictions, "XGBoost Predictions", "--", "red"),
    ],
    "Uber 2014 Trips: XGBoost Predictions vs Test",
)
xgb_mape = mean_absolute_percentage_error(uber2014_test["Count"], xgb_predictions)
print(f"XGBoost MAPE:\t\t{xgb_mape:.2%}")

# %% 5. Random Forest
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
}
rf_model = RandomForestRegressor(random_state=seed)
rf_grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=tscv,
    n_jobs=-1,
    scoring="neg_mean_absolute_percentage_error",
    verbose=1,
)
rf_grid_search.fit(X_train, y_train)
print("Best Random Forest parameters:", rf_grid_search.best_params_)

rf_predictions = rf_grid_search.best_estimator_.predict(X_test)
plot_predictions(
    [
        (uber2014_test.index, uber2014_test["Count"], "Test", "-", "gray"),
        (uber2014_test.index, rf_predictions, "Random Forest Predictions", "--", "green"),
    ],
    "Uber 2014 Trips: Random Forest Predictions vs Test",
)
rf_mape = mean_absolute_percentage_error(uber2014_test["Count"], rf_predictions)
print(f"Random Forest MAPE:\t{rf_mape:.2%}")

# %% 6. Gradient Boosted Regression Tree
gbr_param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
gbr_model = GradientBoostingRegressor(random_state=seed)
gbr_grid_search = GridSearchCV(
    estimator=gbr_model,
    param_grid=gbr_param_grid,
    cv=tscv,
    n_jobs=-1,
    scoring="neg_mean_absolute_percentage_error",
    verbose=1,
)
gbr_grid_search.fit(X_train, y_train)
print("Best GBRT parameters:", gbr_grid_search.best_params_)

gbr_predictions = gbr_grid_search.best_estimator_.predict(X_test)
plot_predictions(
    [
        (uber2014_test.index, uber2014_test["Count"], "Test", "-", "gray"),
        (uber2014_test.index, gbr_predictions, "GBRT Predictions", "--", "orange"),
    ],
    "Uber 2014 Trips: GBRT Predictions vs Test",
)
gbr_mape = mean_absolute_percentage_error(y_test, gbr_predictions)
print(f"GBRT MAPE:\t\t{gbr_mape:.2%}")

# %% 7. All models together
plot_predictions(
    [
        (uber2014_test.index, uber2014_test["Count"], "Test", "-", "gray"),
        (uber2014_test.index, xgb_predictions, "XGBoost Predictions", "--", "red"),
        (uber2014_test.index, gbr_predictions, "GBRT Predictions", "--", "orange"),
        (uber2014_test.index, rf_predictions, "Random Forest Predictions", "--", "green"),
    ],
    "Uber 2014 Trips: All Models Predictions vs Test",
)

# %% 8. Ensemble (inverse-MAPE weighted average)
# The original notebook hard-coded weights [0.368, 0.322, 0.310];
# here they are computed from the MAPEs (very close to those values).
inv = np.array([1 / xgb_mape, 1 / rf_mape, 1 / gbr_mape])
weights = inv / inv.sum()
print("Ensemble weights (XGB, RF, GBRT):", weights.round(3))

ensemble_predictions = (
    weights[0] * xgb_predictions
    + weights[1] * rf_predictions
    + weights[2] * gbr_predictions
)
plot_predictions(
    [
        (uber2014_test.index, uber2014_test["Count"], "Test", "-", "gray"),
        (uber2014_test.index, ensemble_predictions, "Ensemble Predictions", "--", "purple"),
    ],
    "Uber 2014 Trips: Ensemble Predictions vs Test",
)
ensemble_mape = mean_absolute_percentage_error(uber2014_test["Count"], ensemble_predictions)

print(f"XGBoost MAPE:\t\t{xgb_mape:.2%}")
print(f"Random Forest MAPE:\t{rf_mape:.2%}")
print(f"GBRT MAPE:\t\t{gbr_mape:.2%}")
print(f"Ensemble MAPE:\t\t{ensemble_mape:.2%}")
