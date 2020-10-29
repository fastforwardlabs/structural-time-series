import argparse

import kfp
from kfp import dsl

# get host argument

parser = argparse.ArgumentParser()
parser.add_argument('--host')
parser.add_argument('--run_name')
args = parser.parse_args()

# define pipeline componenets

def load_and_preprocess_op():

    return dsl.ContainerOp(
        name='Load and Preprocess Data',
        image='andrewrreed/cffl-sts-image:latest',
        command=['python3', 'kfp/scripts/load_and_preprocess.py'],
        arguments=[],
        file_outputs={
            'data_df': '/usr/src/app/kfp/data/data_df.pkl',
            'train_df': '/usr/src/app/kfp/data/train_df.pkl'
        }
    )

def fit_score_simple_prophet_op(train_df):

    return dsl.ContainerOp(
        name='Fit and Score Simple Prophet Model',
        image='andrewrreed/cffl-sts-image:latest',
        command=['python3', 'kfp/scripts/fit_score_simple_prophet_model.py'],
        arguments=[
            '--train_df', train_df
        ],
        file_outputs={
            'prophet_simple': '/usr/src/app/kfp/data/prophet_simple.csv'
        }
    )

def fit_score_complex_prophet_op(train_df):

    return dsl.ContainerOp(
        name='Fit and Score Complex Prophet Model',
        image='andrewrreed/cffl-sts-image:latest',
        command=['python3', 'kfp/scripts/fit_score_complex_prophet_model.py'],
        arguments=[
            '--train_df', train_df
        ],
        file_outputs={
            'prophet_complex': '/usr/src/app/kfp/data/prophet_complex.csv'
        }
    )

def fit_score_complex_log_prophet_op(train_df):

    return dsl.ContainerOp(
        name='Fit and Score Complex Log Prophet Model',
        image='andrewrreed/cffl-sts-image:latest',
        command=['python3', 'kfp/scripts/fit_score_complex_log_prophet_model.py'],
        arguments=[
            '--train_df', train_df
        ],
        file_outputs={
            'prophet_log_complex': '/usr/src/app/kfp/data/prophet_log_complex.csv'
        }
    )

def evaluate_models_op(prophet_simple, prophet_complex, prophet_log_complex, data_df):

    return dsl.ContainerOp(
        name='Evaluate Both Models',
        image='andrewrreed/cffl-sts-image:latest',
        command=['python3', 'kfp/scripts/evaluate_models.py'],
        arguments=[
            '--prophet_simple', prophet_simple,
            '--prophet_complex', prophet_complex,
            '--prophet_log_complex', prophet_log_complex,
            '--data_df', data_df
        ],
        file_outputs={}
    )

# define kubeflow pipeline

@dsl.pipeline(
    name='CFFL Structural-Time-Series Demo Pipeline',
    description='A demo kubeflow pipeline.'
)
def cffl_sts_pipeline():
    
    _load_and_preprocess_op = load_and_preprocess_op()

    _fit_score_simple_prophet_op = fit_score_simple_prophet_op(
        dsl.InputArgumentPath(_load_and_preprocess_op.outputs['train_df'])
    ).after(_load_and_preprocess_op)

    _fit_score_complex_prophet_op = fit_score_complex_prophet_op(
        dsl.InputArgumentPath(_load_and_preprocess_op.outputs['train_df'])
    ).after(_load_and_preprocess_op)

    _fit_score_complex_log_prophet_op = fit_score_complex_log_prophet_op(
        dsl.InputArgumentPath(_load_and_preprocess_op.outputs['train_df'])
    ).after(_load_and_preprocess_op)

    _evaluate_models_op = evaluate_models_op(
        dsl.InputArgumentPath(_fit_score_simple_prophet_op.outputs['prophet_simple']),
        dsl.InputArgumentPath(_fit_score_complex_prophet_op.outputs['prophet_complex']),
        dsl.InputArgumentPath(_fit_score_complex_log_prophet_op.outputs['prophet_log_complex']),
        dsl.InputArgumentPath(_load_and_preprocess_op.outputs['data_df'])
    ).after(_fit_score_complex_log_prophet_op)

    
if __name__ == '__main__':
    
    # compile and generate compressed kubeflow pipeline file locally
    kfp.compiler.Compiler().compile(cffl_sts_pipeline, 'cffl_sts_pipeline.yaml')

    # create client connection and execute pipeline run

    client = kfp.Client(host=args.host)
    client.create_run_from_pipeline_func(cffl_sts_pipeline,
                                         arguments={},
                                         experiment_name='structural_time_series_demo',
                                         run_name=args.run_name
    )