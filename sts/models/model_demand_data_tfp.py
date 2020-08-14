import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts


#num_days_per_month = np.array(
#  [[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
#   [31, 29, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],  # year with leap day
#   [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
#   [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31]])
#
#month_of_year = tfp.sts.Seasonal(
#  num_seasons=12,
#  num_steps_per_season=num_days_per_month,
#  drift_scale_prior=tfd.LogNormal(loc=-1., scale=0.1),
#  initial_effect_prior=tfd.Normal(loc=0., scale=5.),
#  name='month_of_year')

data = pd.read_csv('demand.csv')
demand = data['demand']
standardized_observations = (demand - demand.mean()) / demand.std()


def model(observed_time_series):
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)

    day_of_week = sts.Seasonal(
        num_seasons=7,
        num_steps_per_season=24,
        observed_time_series=observed_time_series,
        name='day_of_week'
    )
    
    hour_of_day = sts.Seasonal(
        num_seasons=24,
        num_steps_per_season=1,
        observed_time_series=observed_time_series
    )
    
    model = sts.Sum(
        components=[trend, day_of_week, hour_of_day],
        observed_time_series=observed_time_series
    )
    
    return model


demand_model = model(standardized_observations)
joint = demand_model.joint_log_prob(observed_time_series=standardized_observations)

variational_posteriors = sts.build_factored_surrogate_posterior(model=demand_model)



num_variational_steps = 200

optimizer = tf.optimizers.Adam(learning_rate=0.1)

@tf.function(experimental_compile=True)
def train():
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=joint,
        surrogate_posterior=variational_posteriors,
        optimizer=optimizer,
        num_steps=num_variational_steps
    )
    return elbo_loss_curve

elbo_loss_curve = train()