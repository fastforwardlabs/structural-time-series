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

from fbprophet import Prophet


def default_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model


def multiplicative_prophet_model(df):
    model = Prophet(seasonality_mode='multiplicative')
    model.fit(df)
    return model


def seasonal_daily_prophet_model(df):
    model = Prophet(
        daily_seasonality=False,
        yearly_seasonality=20,
        changepoint_prior_scale=0.001
    )
    model.add_seasonality(
        name='winter_weekday',
        period=1,
        fourier_order=12,
        condition_name='winter_weekday'
    )
    model.add_seasonality(
        name='winter_weekend',
        period=1,
        fourier_order=12,
        condition_name='winter_weekend'
    )
    model.add_seasonality(
        name='summer_weekday',
        period=1,
        fourier_order=12,
        condition_name='summer_weekday'
    )
    model.add_seasonality(
        name='summer_weekend',
        period=1,
        fourier_order=12,
        condition_name='summer_weekend'
    )
    model.add_country_holidays(country_name='US')
    df = add_season_weekday_indicators(df)
    model.fit(df)
    return model


def add_season_weekday_indicators(df):
    df['winter_weekday'] = df['ds'].apply(is_winter_weekday)
    df['winter_weekend'] = df['ds'].apply(is_winter_weekend)
    df['summer_weekday'] = df['ds'].apply(is_summer_weekday)
    df['summer_weekend'] = df['ds'].apply(is_summer_weekend)
    return df


def is_winter_weekday(ds):
    condition = (
        (ds.month < 6 or ds.month >= 10)
        and not (ds.day_name() in ['Saturday', 'Sunday'])
    )
    return condition


def is_winter_weekend(ds):
    condition = (
        (ds.month < 6 or ds.month >= 10)
        and (ds.day_name() in ['Saturday', 'Sunday'])
    )
    return condition


def is_summer_weekday(ds):
    condition = (
        (ds.month >= 6 or ds.month < 10)
        and not (ds.day_name() in ['Saturday', 'Sunday'])
    )
    return condition


def is_summer_weekend(ds):
    condition = (
        (ds.month >= 6 or ds.month < 10)
        and (ds.day_name() in ['Saturday', 'Sunday'])
    )
    return condition
