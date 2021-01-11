import os
import pickle
import argparse
from sts.models.prophet import default_prophet_model


def _fit_score_simple_prophet(train_df):

    with open(train_df, 'rb') as f:
        df = pickle.load(f)

    # ## Prophet (Default)
    # FB Prophet model, all default parameters.

    model = default_prophet_model(df)

    future = model.make_future_dataframe(periods = 8760, freq='H')
    forecast = model.predict(future)

    KFP_DIR = 'kfp/data/'

    if not os.path.exists(KFP_DIR):
        os.makedirs(KFP_DIR)

    forecast[['ds', 'yhat']].to_csv(KFP_DIR + 'prophet_simple.csv', index=False)

if __name__ == '__main__':
    print("-----Fitting and Scoring Simple Prophet Model----")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_df')
    args = parser.parse_args()
    _fit_score_simple_prophet(args.train_df)
