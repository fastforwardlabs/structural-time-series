import os

import numpy as np

from sts.data.loader import load_california_electricity_demand
from sts.models.baselines import year_ahead_hourly_forecast


# Load the data

df = load_california_electricity_demand()

# ## Baseline
# Reproduce observed values exactly 52 weeks prior as forecast.

baseline = (
    df
    .sort_values('ds')
    .assign(yhat=year_ahead_hourly_forecast)
)


# ## Write
# Write the forecast values to csv
DIR = 'data/forecasts'

if not os.path.exists(DIR):
    os.makedirs(DIR)

baseline[['ds', 'yhat']].to_csv(DIR + '/baseline.csv', index=False)