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

The probabilistic model and inference algorithm are described in the
[following paper](https://arxiv.org/abs/2403.07657):

```bibtex
@article{
title   = {Scalable Spatiotemporal Prediction with {Bayesian} Neural Fields},
authors = {Saad, Feras and Burnim, Jacob and Carroll, Colin and Patton, Brian and Köster, Urs  and Saurous, Rif A. and Hoffman, Matthew}
journal = {arXiv},
volume  = {2403.07657},
year    = {2024},
doi     = {10.48550/arXiv.2403.07657},
}
```

## Installation

`bayesnf` can be installed from the Python Package Index
([PyPI](https://pypi.org/project/bayesnf/)) using:

```
$ python -m pip install bayesnf
```

The typical install time is 1 minute. This software is tested on Python 3.10
with a standard Debian GNU/Linux setup. The large-scale experiments in
`scripts/` were run using [TPU v3-8 accelerators](https://cloud.google.com/tpu/docs/supported-tpu-configurations#tpu-v3-config).
To run BayesNF locally on medium to large-scale data, a GPU is
required at minimum.

Installation into a virtual environment is highly recommended, using the
following steps:

```
$ python -m venv pyenv
$ source pyenv/bin/activate
$ python -m pip install -U bayesnf
```

The versions of dependencies will depend on the Python version.
Github Actions tests the software using Python 3.10.
If encountering any version issues, please refer to the following file
for the versions of libraries used in the test suite:
[requirements.Python3.10.14.txt](https://github.com/google/bayesnf/blob/main/requirements.Python3.10.14.txt).
These specific versions can be installed into the virtual environment
using the following command:

```
$ python -m pip install -r requirements.Python3.10.14.txt
```


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
