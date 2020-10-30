import os

import numpy as np

from sts.data.loader import load_california_electricity_demand
from sts.models.prophet import default_prophet_model


# Load the training data (through 2018)

df = load_california_electricity_demand(train_only=True)


# ## Prophet (Default)
# FB Prophet model, all default parameters.

model = default_prophet_model(df)

future = model.make_future_dataframe(periods = 8760, freq='H')
forecast = model.predict(future)


# ## Write
# Write the forecast values to csv
DIR = 'data/forecasts/'

if not os.path.exists(DIR):
    os.makedirs(DIR)

forecast[['ds', 'yhat']].to_csv(DIR + 'prophet_simple.csv', index=False)