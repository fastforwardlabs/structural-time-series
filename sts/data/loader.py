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

import os
import json
import requests

import pandas as pd


def load_california_electricity_demand(
        filepath='data/demand.json',
        api_key_env='EIA_API_KEY',
        train_only=False):

    data = read_or_download_data(filepath, api_key_env)

    df = (
        json_to_df(data)
        .rename(columns={0: 'ds', 1: 'y'})
        .assign(ds=utc_to_pst)
        .assign(ds=lambda df: df.ds.dt.tz_localize(None))
        .sort_values('ds')
    )

    if train_only:
        df = remove_2019_and_later(df)

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


def remove_2019_and_later(df):
    return df[df['ds'] < '2019']
