NUM_HOURS_IN_DAY = 24
NUM_DAYS_IN_WEEK = 7


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


def year_ahead_hourly_forecast(df):
    """One year is exactly 52 weeks"""
    return df.shift(periods=52*NUM_HOURS_IN_DAY*NUM_DAYS_IN_WEEK).y


# n-step daily


def day_ahead_daily_forecast(df):
    return df.shift(periods=1).y


def week_ahead_daily_forecast(df):
    return df.shift(periods=NUM_DAYS_IN_WEEK).y


def month_ahead_daily_forecast(df):
    """One month is exactly four weeks"""
    return df.shift(periods=4*NUM_DAYS_IN_WEEK).y


def year_ahead_daily_forecast(df):
    """One year is exactly 52 weeks"""
    return df.shift(periods=52*NUM_DAYS_IN_WEEK).y



# Collect baseline forecasts


def global_mean_forecast(df):
    return df.y.mean()


def hourly_forecasts(df):
    forecasts = df.assign(
        hour_ahead_hourly_forecast=hour_ahead_hourly_forecast,
        day_ahead_hourly_forecast=day_ahead_hourly_forecast,
        week_ahead_hourly_forecast=week_ahead_hourly_forecast,
        month_ahead_hourly_forecast=month_ahead_hourly_forecast,
        year_ahead_hourly_forecast=year_ahead_hourly_forecast
    )
    return forecasts


def daily_forecasts():
    forecasts = df.resample('1D').sum().assign(
        day_ahead_daily_forecast=day_ahead_daily_forecast,
        week_ahead_daily_forecast=week_ahead_daily_forecast,
        month_ahead_daily_forecast=month_ahead_daily_forecast,
        year_ahead_daily_forecast=year_ahead_daily_forecast
    )
    return forecasts