import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("California Electricity Demand Weekly Forecast")

data_loading = st.text("Loading data...")

@st.cache
def load_data():
    return pd.read_csv('data/forecast.csv', parse_dates=['ds'])

data = load_data()
data = data[data.ds.dt.week == 2]
data = data.set_index('ds')

data_loading.text("")

random_samples = data.sample(25, axis='columns')

generating_chart = st.text("Generating chart")
plt.figure()
random_samples.plot(
    legend=False,
    alpha=0.1,
    color='#00828c',
    ylim=[15000, 45000]
)
st.pyplot()
generating_chart.text("")



big_weeks = data.sum() > 4.85e6
filtered_data = data.loc[:, big_weeks]

st.text("The probability of exceeding 4.85 Terrawatts of demand for the full week is {:.2f}%.".format(big_weeks.sum()/1000*100))
st.text("Here are some possible futures in that scenario.")

filtered_data_samples = filtered_data.sample(25, axis='columns')

generating_chart = st.text("Generating chart")
plt.figure()
filtered_data_samples.plot(
    legend=False,
    alpha=0.1,
    color='#00828c',
    ylim=[15000, 45000]
)
st.pyplot()
generating_chart.text("")


spike_weeks = data.max() > 40000
spike_data = data.loc[:, spike_weeks]

st.text("The probability of demand exceeding 40 Gigawatts in an hour at some point this week is {:.2f}%.".format(spike_weeks.sum()/1000*100))
st.text("Here are some possible futures in that scenario.")

spike_data_samples = spike_data.sample(25, axis='columns')

generating_chart = st.text("Generating chart")
plt.figure()
spike_data_samples.plot(
    legend=False,
    alpha=0.1,
    color='#00828c',
    ylim=[15000, 45000]
)
st.pyplot()
generating_chart.text("")