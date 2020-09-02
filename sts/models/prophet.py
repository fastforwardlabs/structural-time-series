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
        yearly_seasonality=5
    )
    model.add_seasonality(
        name='winter_weekday',
        period=24,
        fourier_order=4,
        condition_name='winter_weekday'
    )
    model.add_seasonality(
        name='winter_weekend',
        period=24,
        fourier_order=4,
        condition_name='winter_weekend'
    )
    model.add_seasonality(
        name='summer_weekday',
        period=24,
        fourier_order=4,
        condition_name='summer_weekday'
    )
    model.add_seasonality(
        name='summer_weekend',
        period=24,
        fourier_order=4,
        condition_name='summer_weekend'
    )
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
