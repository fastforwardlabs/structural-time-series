import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.title("California Electricity Demand Weekly Forecast")


# Data loading and selection

start_date, end_date = st.date_input(
  "Select a forecast range",
  [datetime.date(2019,1,1), datetime.date(2020,1,1)] # replace with now, now + 1 year
)

data_loading = st.text("Loading data...")

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("data/forecast.csv", parse_dates=["ds"])
    data = data.set_index("ds")
    return data

@st.cache(allow_output_mutation=True)
def sample(data):
    return data.sample(1, axis="columns")

data = load_data()
data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
data_loading.text("")


# Main forecast plot

mean_forecast = data.mean(axis="columns")

generating_chart = st.text("Generating chart")
line_chart = px.line(
  mean_forecast,
  title="Forecast",
  color_discrete_sequence=["#00828c"]
)
line_chart.update_xaxes(range=[start_date, end_date])
line_chart.update_layout(showlegend=False)
st.plotly_chart(line_chart)
generating_chart.text("")


# Marginal plot of sum of values over interval

data_sum = data.sum()
_min = float(data_sum.min())
_max = float(data_sum.max())

threshold = st.slider(
  "Threshold (Terawatthours)",
  min_value = _min/1e6,
  max_value = _max/1e6
)

prob_exceed = data_sum[data_sum > threshold*1e6].count() / data_sum.count()

st.text(f"""
    The probability of demand exceeding a threshold of {threshold}
    in the selected date range is {prob_exceed}.
""")

hist = px.histogram(
  data_sum[data_sum > threshold*1e6],
  title="Possible total electricity demand levels",
  color_discrete_sequence=["#00828c"]
)
hist.update_xaxes(range=[_min, _max])
hist.update_layout(showlegend=False)
st.plotly_chart(hist)
