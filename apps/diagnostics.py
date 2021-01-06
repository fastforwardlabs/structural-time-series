# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2020
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf

from sts.data.loader import load_california_electricity_demand
from sts.models.baselines import year_ahead_hourly_forecast


FORECAST_DIRECTORY = "data/forecasts"


st.title("California Electricity Demand Model Diagnostics")

# first, load true demand data and forecasts


def read_forecast(filename):
    name = filename.split(".")[0]
    df = (
        pd
        .read_csv(FORECAST_DIRECTORY+"/"+filename)
        .rename(columns={"yhat": name})
        .assign(ds=lambda df: pd.to_datetime(df.ds))
    )
    return df


@st.cache(allow_output_mutation=True)
def load_all_forecasts():
    df = load_california_electricity_demand().sort_values("ds")
    forecast_list = os.listdir(FORECAST_DIRECTORY)
    for f in forecast_list:
        df = df.merge(read_forecast(f), on="ds")
    return df


data_loading = st.text("Loading data...")
df = load_all_forecasts()
data_loading.text("")

df_train = df[df.ds.dt.year < 2019]
df_2018 = df[df.ds.dt.year == 2018]
df_2019 = df[df.ds.dt.year == 2019]

model_names = [x for x in df.columns if x not in ["ds", "y"]]


f"""
## Model comparison
There are {len(model_names)} models. Here is a comparison of their MAPE for select data slices.
We compare a held out test set (2019) to the whole training set through 2018 and also
2018 in isolation. 2018 is included for being one complete period in the training set
of equal length to 2019.
"""


def ape(df):
    return pd.DataFrame({m: np.abs(df.y - df[m]) / df.y for m in model_names})


st.write(
    pd.DataFrame({
        "all training":    ape(df_train).mean().rename("MAPE"),
        "2018 (training)": ape(df_2018).mean().rename("MAPE"),
        "2019  (holdout)": ape(df_2019).mean().rename("MAPE")
    }).transpose()
)


"""
---
Another metric for this kind of time series is [MASE](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error).
We will use a seasonal variant, where the season is defined to be 52 weeks long,
so that years are approximately aligned.
MASE measures error relative to the baseline, so a lower score is better.
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
        "all training":    mase(df_train).mean().rename("MASE"),
        "2018 (training)": mase(df_2018).mean().rename("MASE"),
        "2019  (holdout)": mase(df_2019).mean().rename("MASE")
    }).transpose()
)


"""
---
## Model drill-down
We can compute some more detailed diagnostics for each model individually.
"""
active_model = st.selectbox("Model", model_names)


"""
### The forecast
First, we should see the forecast vs true, observed values.
"""

forecast_chart = px.line(
    df, x='ds', y=['y', active_model],
    color_discrete_sequence=["#ff8300", "#00828c"]
)
forecast_chart.update_xaxes(
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
forecast_chart.update_layout(
    xaxis_title="Datetime (hourly increments)",
    yaxis_title="Demand (Megawatt-hours)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,

    ),
    legend_title_text=""
)
st.plotly_chart(forecast_chart)


"""
---
### Diagnostics
"""

data_set = st.selectbox("Dataset", ['Train', 'Test', 'Combined'])

if data_set == 'Train':
    df = df_train
elif data_set == 'Test':
    df = df_2019


"""
---
Scatter plot of the true values vs forecast values.
This plot will be heavily overplotted, but the overall shape should tell us
whether we are over- or under-predicting.
"""

scatter_chart = go.Figure(data=go.Scatter(
    x=df.y, y=df[active_model],
    mode="markers",
    marker=dict(color="#00828c", opacity=0.2),
))
scatter_chart.update_layout(
    xaxis_title="True demand (Megawatt-hours)",
    yaxis_title="Forecast demand (Megawatt-hours)"
)

st.plotly_chart(scatter_chart)

residuals = (df["y"] - df[active_model]).dropna()


"""
---
Here is the marginal distribution of the residuals.
We expect it to be symmetric, approximately normal, and centered at zero.
"""

residual_chart = px.histogram(
    df, x=residuals, color_discrete_sequence=["#00828c"]
)
residual_chart.update_layout(
    xaxis_title="Residual (true demand - forecast demand) (Megawatt-hours)",
    yaxis_title="Count"
)

st.plotly_chart(residual_chart)


"""
---
The autocorrelation and partial autocorrelation of the residuals.
Since none of our models try to model the error (with autoregressive terms), we may
expect some autocorrelation.
The orange bands represent the 95% confidence interval for the null hypothesis that
there is no (partial) autocorrelation.
Bars outside those bounds indicate high likelihood of autocorrelation.
"""

autocorrelation, conf_intervals = acf(residuals, alpha=0.05, nlags=48)

autocorrelation_df = pd.DataFrame({
    "autocorrelation": autocorrelation,
    # center confidence intervals on zero,
    # so that null hypothesis is zero autocorrelation
    "ci_lower": conf_intervals[:, 0]-autocorrelation,
    "ci_upper": conf_intervals[:, 1]-autocorrelation
})
autocorrelation_chart = px.bar(
    autocorrelation_df,
    x=autocorrelation_df.index,
    y=["autocorrelation", "ci_lower", "ci_upper"],
    color_discrete_sequence=["#00828c", "#ff8300", "#ff8300"],
    barmode="overlay"
)
autocorrelation_chart.update_layout(
    xaxis_title="Timestep (hours)",
    yaxis_title="Autocorrelation",
    showlegend=False
)

st.plotly_chart(autocorrelation_chart)


partial_autocorrelation, partial_conf_intervals = pacf(
    residuals, alpha=0.05, nlags=48
)

partial_autocorrelation_df = pd.DataFrame({
    "partial_autocorrelation": partial_autocorrelation,
    # center confidence intervals on zero,
    # so that null hypothesis is zero partial autocorrelation
    "ci_lower": partial_conf_intervals[:, 0]-partial_autocorrelation,
    "ci_upper": partial_conf_intervals[:, 1]-partial_autocorrelation
})
partial_autocorrelation_chart = px.bar(
    partial_autocorrelation_df,
    x=partial_autocorrelation_df.index,
    y=["partial_autocorrelation", "ci_lower", "ci_upper"],
    color_discrete_sequence=["#00828c", "#ff8300", "#ff8300"],
    barmode='overlay'
)
partial_autocorrelation_chart.update_layout(
    xaxis_title="Timestep (hours)",
    yaxis_title="Partial autocorrelation",
    showlegend=False
)

st.plotly_chart(partial_autocorrelation_chart)
