# bayesnf

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

`bayesnf` can be installed from the Python Package Index (PyPI) using:

```
python -m pip install .
```

Typical install time is 1 minute. This software is tested on Python 3.9.
Experiments were run using TPU accelerators.

## Documentation and Tutorials

Please visit <https://google.github.io/bayesnf>

## Quick start

TODO
