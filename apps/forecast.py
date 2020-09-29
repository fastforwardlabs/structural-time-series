import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.title("California Electricity Demand Forecast")


# Data loading and selection

data_loading = st.text("Loading data...")

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("data/forecast.csv", parse_dates=["ds"])
    data = data.set_index("ds")
    return data

@st.cache(allow_output_mutation=True)
def samples(data):
    return data.sample(10, axis="columns")


data = load_data()

start_date, end_date = st.date_input(
    "Select a forecast range",
    [datetime.date(2020,1,1), datetime.date(2020,12,31)] # replace with now, now + 1 year
)

data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
data_loading.text("")


# Main forecast plot

generating_chart = st.text("Generating chart")
mean_forecast = data.mean(axis="columns")
sample_forecasts = samples(data).reset_index().melt(id_vars='ds')


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
    yaxis_title="Megawatthours"
)
st.plotly_chart(line_chart)
generating_chart.text("")


# Marginal plot of sum of values over interval

data_sum = data.sum()
_min = float(data_sum.min())
_max = float(data_sum.max())


threshold = st.slider(
    "Threshold (Megawatthours)",
    min_value = _min,
    max_value = _max,
    format = "%.2e"
)

prob_exceed = data_sum[data_sum > threshold].count() / data_sum.count()

st.markdown(f"""
    The most likely total demand between {start_date} and {end_date}
    is **{mean_forecast.mean():.2e}** Megawatthours.
""")

st.markdown(f"""
    The probability of the total demand between {start_date} and {end_date}
    being more than {threshold:.2e} Megawatthours
    is **{100*prob_exceed:.1f}**%.
""")

st.markdown("""

""")

hist = px.histogram(
  data_sum[data_sum > threshold],
  title="Possible total electricity demand levels",
  color_discrete_sequence=["#00828c"]
)
hist.update_xaxes(range=[_min, _max])
hist.update_layout(
  showlegend=False,
  xaxis_title="Megawatthours",
  yaxis_title="Count (of 1000 simulated futures)"
)
st.plotly_chart(hist)
