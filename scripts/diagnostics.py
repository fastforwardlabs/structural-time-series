# ## Baseline

br = baseline_results

br.plot(x='ds', y=['y', 'yhat'])

br[(br.ds.dt.year == 2019) & (br.ds.dt.month.isin([8,9,10]))].plot(x='ds', y=['y', 'yhat'])



# ## Prophet

pr = forecast.merge(df, on='ds')

pr.plot(x='ds', y=['y', 'yhat'])

pr[(pr.ds.dt.year == 2019) & (pr.ds.dt.month.isin([8,9,10]))].plot(x='ds', y=['y', 'yhat'])


# ## Prophet with complex seasonality

sr = seasonal_forecast.merge(df, on='ds')

sr.plot(x='ds', y=['y', 'yhat'])

sr[(sr.ds.dt.year == 2019) & (sr.ds.dt.month.isin([8,9,10]))].plot(x='ds', y=['y', 'yhat'])

