import os

import numpy as np

from sts.data.loader import load_california_electricity_demand
from sts.models.prophet import (
    add_season_weekday_indicators,
    seasonal_daily_prophet_model
)


# Load the training data (through 2018)

df = load_california_electricity_demand(train_only=True)

# Log transform the target variable
df['y'] = df.y.apply(np.log)


# ## Prophet (with more complicated seasonality)
# FB Prophet model, splitting intra-day seasonalities into four subgroups:
# - summer weekday
# - summer weekend
# - winter weekday
# - winter weekend

model = seasonal_daily_prophet_model(df)

future = model.make_future_dataframe(periods = 8760, freq='H')
seasonal_future = add_season_weekday_indicators(future)

forecast = model.predict(seasonal_future)

# Reverse the log transform on predictions
forecast['yhat'] = forecast.yhat.apply(np.exp)


# ## Write
# Write the forecast values to csv
DIR = 'data/forecasts/'

if not os.path.exists(DIR):
    os.makedirs(DIR)

forecast[['ds', 'yhat']].to_csv(DIR + 'prophet_complex_log.csv', index=False)