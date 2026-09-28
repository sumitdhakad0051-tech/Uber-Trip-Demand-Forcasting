# Uber Trip Analysis & Demand Forecasting

Analysis of NYC Uber pickups (Apr–Sep 2014) and hourly demand forecasting using
XGBoost, Random Forest, Gradient Boosted Trees, and a weighted ensemble.

## Files
- `uber_trip_baseline.py` – EDA + baseline Random Forest on one month of data
- `uber_demand_forecasting.py` – full pipeline: hourly resampling, seasonal decomposition,
  time-based train/test split, lagged features (24h window), grid search with `TimeSeriesSplit`,
  ensemble
- `requirements.txt`

## Data
Download the Uber TLC FOIL dataset (FiveThirtyEight / Kaggle "Uber Pickups in New York City")
and place `uber-raw-data-apr14.csv` … `uber-raw-data-sep14.csv` in a `data/` folder
(the baseline script reads `uber-raw-data-apr14.csv` from the working directory).

## Run
```bash
pip install -r requirements.txt
python uber_demand_forecasting.py
```
Files use `# %%` cell markers, so they also run cell-by-cell in VS Code / Jupyter.

## Results (test: 2014-09-15 → 2014-09-30)
| Model | MAPE |
|---|---|
| XGBoost | 8.37% |
| Random Forest | 9.61% |
| GBRT | 10.02% |
| Ensemble | 8.60% |
