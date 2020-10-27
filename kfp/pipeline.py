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
        command='python3 kfp/scripts/load_and_preprocess.py',
        arguments=[],
        file_outputs={
            'data_df': 'kfp/data/data_df.pkl'
        }
    )

def fit_score_simple_prophet_op(data_df):

    return dsl.ContainerOP(
        name='Fit and Score Simple Prophet Model',
        image='andrewrreed/cffl-sts-image:latest',
        command='python3 kfp/scripts/fit_score_simple_prophet_model.py',
        arguments=[
            '--data_df', data_df
        ],
        file_outputs={
            'prophet_simple': 'kfp/data/prophet_simple.csv'
        }
    )

def fit_score_complex_prophet_op(data_df):

    return dsl.ContainerOP(
        name='Fit and Score Complex Prophet Model',
        image='andrewrreed/cffl-sts-image:latest',
        command='python3 kfp/scripts/fit_score_complex_prophet_model.py',
        arguments=[
            '--data_df', data_df
        ],
        file_outputs={
            'prophet_complex': 'kfp/data/prophet_complex.csv'
        }
    )

# define kubeflow pipeline

@dsl.pipeline(
    name='CFFL Structural-Time-Series Demo Pipeline',
    description='A demo kubeflow pipeline.'
)
def cffl_sts_pipeline():
    
    _load_and_preprocess_op = load_and_preprocess_op()

    _fit_score_simple_prophet_op = fit_score_simple_prophet_op(
        dsl.InputArgumentPath(_load_and_preprocess_op.outputs['data_df'])
    ).after(_load_and_preprocess_op)

    _fit_score_complex_prophet_op = fit_score_complex_prophet_op(
        dsl.InputArgumentPath(_load_and_preprocess_op.outputs['data_df'])
    ).after(_load_and_preprocess_op)


# create client connection and execute pipeline run

client = kfp.Client(host=args.host)
client.create_run_from_pipeline_func(cffl_sts_pipeline)