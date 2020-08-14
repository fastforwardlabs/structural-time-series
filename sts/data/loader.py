import os
import json
import requests

import pandas as pd


def load_california_electricity_demand(
    filepath='data/demand.json',
    api_key_env='EIA_API_KEY'):
    
    data = read_or_download_data(filepath, api_key_env)
        
    df = (
        json_to_df(data)
        .rename(columns={0: 'ds', 1: 'y'})
        .assign(ds=utc_to_pst)
        .assign(ds=lambda df: df.ds.dt.tz_localize(None))
    )
    return df


def read_or_download_data(filepath, api_key_env):

    if os.path.exists(filepath):
        data = read_json(filepath)
    else:
        api_key = try_get_env(api_key_env)
        response_json = fetch_california_demand(api_key)  
        write_json(response_json, filepath)
        data = read_json(filepath)

    return data


def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def write_json(data, filepath):
    with open(filepath, 'w') as file:
        json.dump(data, file)


def try_get_env(api_key_env):
    env = os.getenv(api_key_env)
    if env:
        return env
    else:
        print('Please provide a valid EIA_API_KEY environment variable.')
        return None


def fetch_california_demand(api_key):
    r = requests.get(
        'http://api.eia.gov/series',
        params={
            'api_key': api_key,
            'series_id': 'EBA.CAL-ALL.D.H',
            'out': 'json'
        }
    )
    return r.json()


def json_to_df(data):
    df = pd.DataFrame(data['series'][0]['data'])
    return df


def utc_to_pst(df):
    """
    Convert from UTC to PST.
    PST is always UTC -8 hours: it ignores daylight savings.
    """
    pst = (
        pd
        .to_datetime(df['ds'])
        .subtract(pd.Timedelta('8 hours'))
    )
    return pst







    