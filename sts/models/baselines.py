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
