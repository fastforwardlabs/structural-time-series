import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sts.data.loader import load_california_electricity_demand
from sts.models.baselines import year_ahead_hourly_forecast

FORECAST_DIRECTORY = 'data/forecasts'


st.title("California Electricity Demand Model Diagnostics")

data_loading = st.text("Loading data...")

"""
## Model comparison
"""

df = load_california_electricity_demand().sort_values('ds')

def read_forecast(filename):
    name = filename.split('.')[0]
    df = (
        pd
        .read_csv(FORECAST_DIRECTORY+'/'+filename)
        .rename(columns={'yhat': name})
        .assign(ds=lambda df: pd.to_datetime(df.ds))
    )
    return df


# ! Imperative, do stuff code.
forecast_list = os.listdir(FORECAST_DIRECTORY)

for f in forecast_list:
    df = df.merge(read_forecast(f), on='ds')


data_loading.text("")

model_names = [x for x in df.columns if x not in ['ds', 'y']]

df_train = df[df.ds.dt.year < 2019]
df_2018 = df[df.ds.dt.year == 2018]
df_2019 = df[df.ds.dt.year == 2019]

f"""
There are {len(model_names)} models. Here is a basic comparison of their MAPE.
"""

def ape(df):
    return pd.DataFrame({m: np.abs(df.y - df[m]) / df.y for m in model_names})

st.write(
    pd.DataFrame({
        'all training':    ape(df_train).mean().rename('MAPE'),
        '2018 (training)': ape(df_2018).mean().rename('MAPE'),
        '2019  (holdout)': ape(df_2019).mean().rename('MAPE')
    }).transpose()
)

"""
A better metric for this kind of time series is MASE.
We will use a seasonal variant, where the season is defined to be 52 weeks long,
so that years are approximately aligned.
"""

def mase_denominator(df):
    naive_forecast = year_ahead_hourly_forecast(df)
    denom = np.sum(
      np.abs((naive_forecast - df.y).dropna())
    ) / len(naive_forecast.dropna())
    return denom

denom = mase_denominator(df_train)

def mase(df):
    return pd.DataFrame({m: np.abs(df.y - df[m]) / denom for m in model_names})

st.write(
    pd.DataFrame({
        'all training':    mase(df_train).mean().rename('MASE'),
        '2018 (training)': mase(df_2018).mean().rename('MASE'),
        '2019  (holdout)': mase(df_2019).mean().rename('MASE')
    }).transpose()
)


"""
## Model drill-down
We can compute some more detailed diagnostics for each model individually.
"""

active_model = st.selectbox('Model', model_names)

"""
### The forecast
First, we should see the forecast vs actuals.
"""

fig = px.line(
    df, x='ds', y=['y', active_model],
    color_discrete_sequence=["#ff8300","#00828c"]
)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,

    ),
    legend_title_text=""
)
st.plotly_chart(fig)


"""
### Diagnostics
"""

"Scatter plot of the true values (x) vs forecast values (y)."

st.plotly_chart(go.Figure(data=go.Scatter(
    x=df.y, y=df[active_model],
    mode='markers',
    marker=dict(color="#00828c")
)))


residuals = (df['y'] - df[active_model])

"Distribution of the residuals."

st.plotly_chart(px.histogram(df, x=residuals, color_discrete_sequence=["#00828c"]))


"Autocorrelation of the residuals."
plot_acf(x=residuals)
st.pyplot()

plot_pacf(x=residuals)
st.pyplot()