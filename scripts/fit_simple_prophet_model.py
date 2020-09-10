import os

import numpy as np

from sts.data.loader import load_california_electricity_demand
from sts.models.prophet import default_prophet_model


# Load the data

df = load_california_electricity_demand()


# ## Prophet (Default)
# FB Prophet model, all default parameters.

train_df = (
    df
    [df['ds'] < '2019']
    .sort_values('ds')
    .reset_index(drop=True)
)

model = default_prophet_model(train_df)

future = model.make_future_dataframe(periods = 8760, freq='H')
forecast = model.predict(future)

# ## Write
# Write the forecast values to csv
DIR = 'data/forecasts'

if not os.path.exists(DIR):
    os.makedirs(DIR)

forecast[['ds', 'yhat']].to_csv(DIR + '/prophet_simple.csv', index=False)