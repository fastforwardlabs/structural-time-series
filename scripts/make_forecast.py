from sts.models.prophet import (
    add_season_weekday_indicators,
    seasonal_daily_prophet_model
)
from sts.data.loader import load_california_electricity_demand


# Load the data

df = load_california_electricity_demand()

# Use pre-2019 to train

train_df = (
    df
    [df['ds'] < '2019']
    .sort_values('ds')
    .reset_index(drop=True)
)


# Fit best current model

model = seasonal_daily_prophet_model(train_df)


# Make predictions

future = add_season_weekday_indicators(
    model.make_future_dataframe(periods = 8760, freq='H')
)

forecast = model.predict(future)

samples = model.predictive_samples(future)

predictions = samples['yhat']

prediction_df = (
    future
    .merge(pd.DataFrame(predictions), left_index=True, right_index=True)
    .drop(['winter_weekday', 'winter_weekend', 'summer_weekday', 'summer_weekend'],
          axis='columns')
    [future.ds.dt.year == 2019]
)


# Save predictions
prediction_df.to_csv('data/forecast.csv', index=False)