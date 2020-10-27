import argparse

import kfp
from kfp import dsl

# get host argument

parser = argparse.ArgumentParser()
parser.add_argument('--host')
args = parser.parse_args()

# define pipeline componenets

def load_and_preprocess_op():

    return dsl.ContainerOP(
        name='Load and Preprocess Data',
        image='andrewrreed/cffl-sts-image:latest',
        arguments=['python3 kfp/load_and_preprocess.py'],
        file_outputs={
            'data_df': 'kfp/data/data_df.pkl'
        }
    )

# define kubeflow pipeline

@dsl.pipeline(
    name='CFFL Structural-Time-Series Demo Pipeline',
    description='A demo kubeflow pipeline.'
)
def cffl_sts_pipeline():
    
    _load_and_preprocess_op = load_and_preprocess_op()



# create client connection and execute pipeline run

client = kfp.Client(host=args.host)
client.create_run_from_pipeline_func(boston_pipeline, arguments={})