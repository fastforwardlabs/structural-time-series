import os
import pickle
from sts.data.loader import load_california_electricity_demand

def _load_and_preprocess():

    # Load the training data (through 2018)
    df = load_california_electricity_demand(train_only=True)

    KFP_DIR = 'kfp/data'
    with open(data_path+'/data_df.pkl', 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    print("-----Loading and Preprocessing Data----")
    _load_and_preprocess()