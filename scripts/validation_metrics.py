import datetime

import numpy as np
import pandas as pd

from sts.models.baselines import year_ahead_hourly_forecast
from sts.models.prophet import (
    add_season_weekday_indicators,
    seasonal_daily_prophet_model
)
from sts.data.loader import load_california_electricity_demand


# Load all available data for training

df = load_california_electricity_demand()

# Restrict to pre-2020 for evaluation on 2020
train_df = df[df.ds.dt.year < 2020]

# Take log transform for fully multiplicative model
train_df['y'] = train_df.y.apply(np.log)


# Fit best current model

model = seasonal_daily_prophet_model(train_df)


# Make predictions for one year ahead of most recent training data

future = add_season_weekday_indicators(
    model.make_future_dataframe(periods = 24*365, freq='H')
)

forecast = model.predict(future)

# Reverse log transform
forecast['yhat'] = np.exp(forecast['yhat'])
train_df['y'] = np.exp(train_df['y'])

predictions = (
    forecast[['ds', 'yhat']]
    .merge(df, on='ds')
)
predictions = predictions[predictions.ds.dt.year == 2020]

# ### MAPE
mape = (np.abs(predictions.y - predictions.yhat) / predictions.y).mean()

# Let's compare this to the MAPE of the seasonal naive model
naive_df = df.copy()
naive_df['yhat'] = year_ahead_hourly_forecast(naive_df)
naive_df = naive_df[naive_df.ds.dt.year == 2020]
naive_mape = (np.abs(naive_df.yhat - naive_df.y) / naive_df.y).mean()

# ### MASE
# Note, we have trained on a larger data set than we did for model selection.
# As such, this MASE cannot be compared to the MASEs listed in the diagnostic
# app. It's a measure of performance relative to the baseline on the new
# training set of all data before 2020.
# (The deep reason here is that time series are non-iid, and as such, we
# must make train/dev/validation splits along choronological lines.
# An unfortunate artefact of this is never having the metrics for the exact
# model we deploy.)

naive_forecast = year_ahead_hourly_forecast(train_df)
denom = (
    np.sum(np.abs((naive_forecast - train_df.y).dropna()))
    / len(naive_forecast.dropna())
)
mase = (np.abs(predictions.y - predictions.yhat) / denom).mean()

print(f"The MAPE of our best performing model is: {mape}")
print(f"The MAPE of the seasonal naive baseline: {naive_mape}")
print(f"The MASE of the best performing model is: {mase}")