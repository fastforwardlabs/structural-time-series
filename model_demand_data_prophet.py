import pandas as pd
import seaborn as sns
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics

from data import load_california_electricity_demand

df = (
    load_california_electricity_demand('demand.json')
    .assign(ds=lambda df: df.ds.dt.strftime('%Y-%m-%d %H:%M:%S'))
)


# ## With additive seasonality 

additive_model = Prophet(seasonality_mode='additive')
additive_model.fit(df)
additive_cv = cross_validation(
    additive_model,
    initial='720 days',
    period='180 days',
    horizon='365 days'
)
additive_perf = performance_metrics(additive_cv)


# ## With multiplicative seasonality

multiplicative_model = Prophet(seasonality_mode='additive')
multiplicative_model.fit(df)
multiplicative_cv = cross_validation(
    multiplicative_model,
    initial='720 days',
    period='180 days',
    horizon='365 days'
)
multiplicative_perf = performance_metrics(multiplicative_cv)


# ## Get samples

m = additive_model

future = m.make_future_dataframe(periods = 8760, freq='H')
forecast = m.predict(future)

m.plot(forecast)

m.plot_components(forecast)

posterior_samples = m.predictive_samples(future)
predictions = posterior_samples['yhat']