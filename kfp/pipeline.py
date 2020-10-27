import argparse

import kfp
from kfp import dsl

def load_and_preprocess_op():

    return dsl.ContainerOP(
        name='Load and Preprocess Data',
        image='andrewrreed/cffl-sts-image:latest',
        arguments=['python3 kfp/load_and_preprocess.py'],
        file_outputs={
            'data_df': 'kfp/data/data_df.pkl'
        }
    )
