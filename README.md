# bayesnf

[![Unittests](https://github.com/google/bayesnf/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google/bayesnf/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/bayesnf.svg)](https://badge.fury.io/py/bayesnf)

*This is not an officially supported Google product.*


Spatially referenced time series (i.e., spatiotemporal) datasets are ubiquitous in scientific, engineering, and business-intelligence applications. Good models of spatial processes that vary over time must be both flexible enough to capture complex statistical dynamics and scalable enough to handle large datasets. This work presents the Bayesian Neural Field - a novel spatiotemporal modeling method that integrates hierarchical probabilistic modeling for accurate uncertainty estimation with deep neural networks for high-capacity function approximation.

Bayesian Neural Fields infer joint probability distributions over field values at arbitrary points in time and space, which makes the model suitable for many data-analysis tasks including spatial interpolation, temporal forecasting, and variography. Posterior inference is conducted using variationally learned surrogates trained via mini-batch stochastic gradient descent for handling large-scale data.

## Installation

`bayesnf` may be installed using

```
python -m pip install .
```

Typical install time is 1 minute.

The library directly depends on the following software (which will be automatically installed as well):

```
flax
jax>=0.4.6
jaxtyping
numpy
optax
pandas
tensorflow-probability[jax]>=0.19.0
```

It has been tested on Python 3.9. Experiments were run using TPU accelerators.

## Quick start

TODO(colcarroll)