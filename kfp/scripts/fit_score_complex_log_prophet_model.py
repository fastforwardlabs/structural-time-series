import os
import pickle
import argparse
import numpy as np
from sts.models.prophet import (
    add_season_weekday_indicators,
    seasonal_daily_prophet_model
)

def _fit_score_complex_log_prophet(train_df):

    with open(train_df, 'rb') as f:
        df = pickle.load(f)

    # Log transform the target variable
    df['y'] = df.y.apply(np.log)

    # ## Prophet (with more complicated seasonality)
    # FB Prophet model, splitting intra-day seasonalities into four subgroups:
    # - summer weekday
    # - summer weekend
    # - winter weekday
    # - winter weekend

    model = seasonal_daily_prophet_model(df)

    future = model.make_future_dataframe(periods = 8760, freq='H')
    seasonal_future = add_season_weekday_indicators(future)

    forecast = model.predict(seasonal_future)

    # Reverse the log transform on predictions
    forecast['yhat'] = forecast.yhat.apply(np.exp)

    KFP_DIR = 'kfp/data/'
    
    if not os.path.exists(KFP_DIR):
        os.makedirs(KFP_DIR)

    forecast[['ds', 'yhat']].to_csv(KFP_DIR + 'prophet_log_complex.csv', index=False)

if __name__ == '__main__':
    print("-----Fitting and Scoring Complex Log Prophet Model----")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_df')
    args = parser.parse_args()
    _fit_score_complex_log_prophet(args.train_df)
