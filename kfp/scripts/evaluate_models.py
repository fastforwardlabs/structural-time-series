import datetime
import pickle
import numpy as np
import pandas as pd
import argparse

def _evaluate_models(prophet_simple, prophet_complex, prophet_log_complex, data_df):

    # load forecasts and actuals

    with open(prophet_simple, 'rb') as f:
        simple_forecast = pd.read_csv(f)

    with open(prophet_complex, 'rb') as f:
        complex_forecast = pd.read_csv(f)

    with open(prophet_log_complex, 'rb') as f:
        complex_log_forecast = pd.read_csv(f)
    
    with open(data_df, 'rb') as f:
        df = pickle.load(f)


    # calcualte MAPE for each forecast

    forecasts = ['simple_forecast', 'complex_forecast', 'complex_log_forecast']

    metrics = {}
    for forecast in forecasts:

        forecast_df = eval(forecast)
        forecast_df.ds = forecast_df.ds.astype('datetime64[ns]')
        predictions = pd.merge(forecast_df, df, on='ds')

        predictions = predictions[predictions.ds.dt.year == 2020]

        mape = (np.abs(predictions.y - predictions.yhat) / predictions.y).mean()

        metrics[forecast] = round(mape, 3)

    print(f'Mean Absolute Percentage Error (MAPE): \n\n {metrics} \n')
    print(f'Deploying {min(metrics)} because it has lower error.')

if __name__ == '__main__':
    print("-----Evaluating Forecasts----")
    parser = argparse.ArgumentParser()
    parser.add_argument('--prophet_simple')
    parser.add_argument('--prophet_complex')
    parser.add_argument('--prophet_log_complex')
    parser.add_argument('--data_df')
    args = parser.parse_args()
    _evaluate_models(
        args.prophet_simple,
        args.prophet_complex,
        args.prophet_log_complex,
        args.data_df
    )