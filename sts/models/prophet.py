from fbprophet import Prophet

def default_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

def multiplicative_prophet_model(df):
    model = Prophet(seasonality_mode='multiplicative')
    model.fit(df)
    return model