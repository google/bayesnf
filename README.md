# Bayesian Neural Fields for Spatiotemporal Prediction

[![Unittests](https://github.com/google/bayesnf/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google/bayesnf/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/bayesnf.svg)](https://badge.fury.io/py/bayesnf)

*This is not an officially supported Google product.*

Spatially referenced time series (i.e., spatiotemporal) datasets are
ubiquitous in scientific, engineering, and business-intelligence
applications. This repository contains an implementation of the Bayesian
Neural Field (BayesNF) a novel spatiotemporal modeling method that
integrates hierarchical probabilistic modeling for accurate uncertainty
estimation with deep neural networks for high-capacity function
approximation.

Bayesian Neural Fields infer joint probability distributions over field
values at arbitrary points in time and space, which makes the model
suitable for many data-analysis tasks including spatial interpolation,
temporal forecasting, and variography. Posterior inference is conducted
using variationally learned surrogates trained via mini-batch stochastic
gradient descent for handling large-scale data.

## Installation

`bayesnf` can be installed from the Python Package Index
([PyPI](https://pypi.org/project/bayesnf/)) using:

```
python -m pip install bayesnf
```

The typical install time is 1 minute. This software is tested on Python 3.9
with a standard Debian GNU/Linux setup. The large-scale experiments in
`scripts/` were run using TPU v3-8 accelerators. For running BayesNF
locally on medium to large-scale data, a GPU is required at minimum.

## Documentation and Tutorials

Please visit <https://google.github.io/bayesnf>

## Quick start

```python

# Load a dataframe with "long" format spatiotemporal data.
df_train = pd.read_csv('chickenpox.5.train.csv',
  index_col=0, parse_dates=['datetime'])

# Build a BayesianNeuralFieldEstimator
model = BayesianNeuralFieldMAP(
  width=256,
  depth=2,
  freq='W',
  seasonality_periods=['M', 'Y'],
  num_seasonal_harmonics=[2, 10],
  feature_cols=['datetime', 'latitude', 'longitude'],
  target_col='chickenpox',
  observation_model='NORMAL',
  timetype='index',
  standardize=['latitude', 'longitude'],
  interactions=[(0, 1), (0, 2), (1, 2)])

# Fit the model.
model = model.fit(
  df_train,
  seed=jax.random.PRNGKey(0),
  ensemble_size=ensemble_size,
  num_epochs=num_epochs)

# Make predictions of means and quantiles on test data.
df_test = pd.read_csv('chickenpox.5.test.csv',
  index_col=0, parse_dates=['datetime'])

yhat, yhat_quantiles = model.predict(df_test, quantiles=(0.025, 0.5, 0.975))
```
