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
sr['y'] = sr.y.apply(np.exp)
sr['yhat'] = sr.yhat.apply(np.exp)

sr.plot(x='ds', y=['y', 'yhat'])

sr[(sr.ds.dt.year == 2019) & (sr.ds.dt.month.isin([8,9,10]))].plot(x='ds', y=['y', 'yhat'])

np.abs((sr.y - sr.yhat) / sr.y).mean()

def plot_autocorrelation(df):
    fig, ax = plt.subplots(2, 1)
    ac = df.y - df.yhat
    plot_acf(x=ac, ax=ax[0])
    plot_pacf(x=ac, ax=ax[1])
    plt.plot()

plot_autocorrelation(sr)