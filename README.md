# Structural Time Series

Applying Bayesian STS to California hourly electricity demand data.

Data source: [US Energy Information Administration](https://www.eia.gov/opendata/qb.php?category=3389936&sdid=EBA.CAL-ALL.D.H)

## Setup

In CML or CDSW, inside a Python 3 session, run `!pip3 install -r requirements.txt' to install dependencies.

The primary entry point to the project is `scripts/evaluate.py`.
Data dependencies will be downloaded on first use.
The evaluation script fits a naive hourly baseline model (forecast what happened 52 weeks ago) and a default hourly Prophet model on pre-2019 data.
Forecasts are made on 2019 data, and MAPE is reported for the whole calendar year of 2019.

### TO FIX

- [ ] Prophet strictly requires that columns are named `ds` and `y`. We have embraced this convention and coupled to it elsewhere in the codebase. Would be good to decouple from naming.
- [ ] The `app.R` script surfaces a simple brush-to-zoom timeseries viewer as a Shiny app. This should be an Application.
- [ ] `app.R` looks for a csv file that no longer exists under the most recent data fetching/loading structure.
- [ ] The data fetching util should automatically create the `data` directory if it doesn't exist. Use [pathlib](https://docs.python.org/3/library/pathlib.html).