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
        yearly_seasonality=20
    )
    model.add_seasonality(
        name='winter_weekday',
        period=1,
        fourier_order=12,
        condition_name='winter_weekday',
        prior_scale=25
    )
    model.add_seasonality(
        name='winter_weekend',
        period=1,
        fourier_order=12,
        condition_name='winter_weekend',
        prior_scale=25
    )
    model.add_seasonality(
        name='summer_weekday',
        period=1,
        fourier_order=12,
        condition_name='summer_weekday',
        prior_scale=25
    )
    model.add_seasonality(
        name='summer_weekend',
        period=1,
        fourier_order=12,
        condition_name='summer_weekend',
        prior_scale=25
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
