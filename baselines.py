import pandas as pd
import numpy as np

from data import load_california_electricity_demand


NUM_HOURS_IN_DAY = 24
NUM_DAYS_IN_WEEK = 7

# Load the data

df = load_california_electricity_demand()

# Separate off the year 2020 for validation

train_df = df[df['ds'] < '2020'].sort_values('ds').reset_index(drop=True).set_index('ds')


# Define some baseline forecasts

# n-step ahead

# n-step hourly

def hour_ahead_hourly_forecast(df):
    return df.shift(periods=1).y

def day_ahead_hourly_forecast(df):
    return df.shift(periods=24).y

def week_ahead_hourly_forecast(df):
    return df.shift(periods=NUM_HOURS_IN_DAY*NUM_DAYS_IN_WEEK).y

def month_ahead_hourly_forecast(df):
    """One month is exactly four weeks"""
    return df.shift(periods=4*NUM_HOURS_IN_DAY*NUM_DAYS_IN_WEEK).y

# n-step daily

def day_ahead_daily_forecast(df):
    return df.shift(periods=1).y

def week_ahead_daily_forecast(df):
    return df.shift(periods=NUM_DAYS_IN_WEEK).y

def month_ahead_daily_forecast(df):
    """One month is exactly four weeks"""
    return df.shift(periods=4*NUM_DAYS_IN_WEEK).y


# Make baseline forecasts

global_mean_forecast = train_df.assign(forecast=lambda df: df.y.mean())

hourly_forecasts = train_df.assign(
    hour_ahead_hourly_forecast=hour_ahead_hourly_forecast,
    day_ahead_hourly_forecast=day_ahead_hourly_forecast,
    week_ahead_hourly_forecast=week_ahead_hourly_forecast,
    month_ahead_hourly_forecast=month_ahead_hourly_forecast
)

daily_forecasts = train_df.resample('1D').sum().assign(
    day_ahead_daily_forecast=day_ahead_daily_forecast,
    week_ahead_daily_forecast=week_ahead_daily_forecast,
    month_ahead_daily_forecast=month_ahead_daily_forecast
)