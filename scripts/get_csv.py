from sts.data.loader import load_california_electricity_demand

# This will load or download the data as json, and write it to csv.
df = load_california_electricity_demand('data/demand.json')
df.to_csv('data/demand.csv')