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

import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


N_SAMPLES = 10

st.title("California Electricity Demand Forecast")


# Data loading and selection

data_loading = st.text("Loading data...")


@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("data/forecast.csv", parse_dates=["ds"])
    data = data.set_index("ds")
    return data


data = load_data()

st.markdown("""
    The forecast is generated for one year ahead of the most recent
    observation. Please select the range of interest over which to view and
    filter samples from the forecast distribution.
""")

start_date, end_date = st.date_input(
    "Select a forecast range",
    [data.index.min().date(), data.index.max().date()]
)

subset = data[(data.index.date >= start_date) &
              (data.index.date <= end_date)].copy()
data_loading.text("")


@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def samples(df):
    return df.sample(N_SAMPLES, axis="columns").reset_index().melt(id_vars='ds')


@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def mean(df):
    return df.mean(axis="columns")

# Main forecast plot


st.markdown(f"""
    The chart below shows the mean forecast (based on 1000 samples),
    and {N_SAMPLES} individual samples, which can be thought of as
    "possible futures".
""")

generating_chart = st.text("Generating chart")
mean_forecast = mean(subset)
sample_forecasts = samples(subset)


line_chart = px.line(
    sample_forecasts,
    x='ds',
    y='value',
    line_group='variable',
    color_discrete_sequence=["rgba(0,130,140,0.1)"],

)
line_chart.add_scatter(
    x=mean_forecast.index,
    y=mean_forecast,
    mode='lines',
    marker=dict(color="rgba(0,130,140,1)")
)
line_chart.update_xaxes(range=[start_date, end_date])
line_chart.update_layout(
    showlegend=False,
    xaxis_title="Datetime (hourly increments)",
    yaxis_title="Megawatt-hours"
)
st.plotly_chart(line_chart)
generating_chart.text("")


# Marginal plot of sum of values over interval

data_sum = subset.sum()
_min = float(data_sum.min())
_max = float(data_sum.max())

st.markdown(f"""
    The mean estimate of the aggregate demand from {start_date} to {end_date}
    is **{data_sum.mean():.2e}** Megawatt-hours.
""")

st.markdown("""
    We can assess the probability of exceeding a given aggregate demand over
    the selected period. Choose the threshold of interest below.
""")

threshold = st.slider(
    "Threshold (Megawatt-hours)",
    min_value=_min,
    max_value=_max,
    format="%.2e"
)

prob_exceed = data_sum[data_sum > threshold].count() / data_sum.count()

st.markdown(f"""
    The probability of the aggregate demand between {start_date} and {end_date}
    being more than {threshold:.2e} Megawatt-hours is
    **{100*prob_exceed:.1f}**%.
""")

st.markdown("""
    The histogram below shows the probability distribution of possible
    aggregate demands, cut off at the threshold selected.
    The higher the count for a given demand, the more likely that future is.
""")

hist = px.histogram(
    data_sum[data_sum > threshold],
    title="Possible total electricity demand levels",
    color_discrete_sequence=["#00828c"]
)
hist.update_xaxes(range=[_min, _max])
hist.update_layout(
    showlegend=False,
    xaxis_title="Megawatt-hours",
    yaxis_title="Count (of 1000 simulated futures)"
)
st.plotly_chart(hist)
