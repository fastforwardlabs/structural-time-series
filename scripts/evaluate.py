import numpy as np

from sts.models.baselines import year_ahead_hourly_forecast
from sts.models.prophet import default_prophet_model
from sts.data.loader import load_california_electricity_demand


# Load the data

df = load_california_electricity_demand()


# ## Baseline
# Reproduce observed values exactly 52 weeks prior as forecast.

baseline_results = (
    df
    .sort_values('ds')
    .assign(yhat=year_ahead_hourly_forecast)
    [(df['ds'] >= '2019') & (df['ds'] < '2020')]
)

baseline_results['ape'] = np.abs(
    (baseline_results.y - baseline_results.yhat) / baseline_results.y
)

baseline_mape = baseline_results.ape.mean()


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

prophet_results = (
    df
    [(df['ds'] >= '2019') & (df['ds'] < '2020')]
    .merge(forecast, on='ds')
)

prophet_results['ape'] = np.abs(
    (prophet_results.y - prophet_results.yhat) / prophet_results.y
)

prophet_mape = prophet_results.ape.mean()


# ## Compare

print(f'Baseline MAPE over 2019: {baseline_mape:.3f}')
print(f'Baseline MAPE over 2019: {prophet_mape:.3f}')

# There are problems with MAPE.
# Should investigate MASE (https://robjhyndman.com/papers/mase.pdf)