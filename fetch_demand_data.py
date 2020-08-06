import os
import requests

import pandas as pd
import seaborn as sns

EIA_API_KEY = os.getenv('EIA_API_KEY')

r = requests.get(
    'http://api.eia.gov/series',
    params={
        'api_key': EIA_API_KEY,
        'series_id': 'EBA.CAL-ALL.D.H',
        'out': 'json'
    }
)

data = r.json()['series'][0]['data']

demand = (
    pd.DataFrame(data)
      .rename(columns={0: 'datetime', 1: 'demand'})
      .assign(
          datetime=lambda x: pd.to_datetime(x['datetime'])
      )
)

demand.to_csv('demand.csv', index=False)