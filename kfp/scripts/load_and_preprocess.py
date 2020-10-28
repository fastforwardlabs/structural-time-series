import os
import pickle
from sts.data.loader import load_california_electricity_demand, remove_2019_and_later

def _load_and_preprocess():

    # Load the training data [and drop 2015 due to memory error]
    df = load_california_electricity_demand(train_only=False)
    df = df[df.ds.dt.year > 2016]

    # Restrict to pre-2020 for evaluation on 2020
    train_df = df[df.ds.dt.year < 2020]

    KFP_DIR = 'kfp/data/'

    if not os.path.exists(KFP_DIR):
        os.makedirs(KFP_DIR)

    with open(KFP_DIR+'train_df.pkl', 'wb') as f:
        pickle.dump(train_df, f)

    with open(KFP_DIR+'data_df.pkl', 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    print("-----Loading and Preprocessing Data----")
    _load_and_preprocess()
