import os
import pickle
import argparse
from sts.models.prophet import (
    add_season_weekday_indicators,
    seasonal_daily_prophet_model
)

def _fit_score_complex_prophet(train_df):

    with open(train_df, 'rb') as f:
        df = pickle.load(f)

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

    KFP_DIR = 'kfp/data/'
    
    if not os.path.exists(KFP_DIR):
        os.makedirs(KFP_DIR)

    forecast[['ds', 'yhat']].to_csv(KFP_DIR + 'prophet_complex.csv', index=False)

if __name__ == '__main__':
    print("-----Fitting and Scoring Complex Prophet Model----")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_df')
    args = parser.parse_args()
    _fit_score_complex_prophet(args.train_df)
