# Copyright 2024 The bayesnf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for spatiotemporal.py."""
from bayesnf import spatiotemporal
import numpy as np
import pytest


@pytest.mark.parametrize(
    "seasonality, freq, expected",
    [
        ("Y", "Y", 1),
        ("Q", "Q", 1),
        ("Y", "Q", 4),
        ("M", "h", 730.5),
        ("Q", "M", 3),
        ("Y", "M", 12),
        ("M", "D", 30.4375),
        ("min", "s", 60),
        ("h", "s", 3600),
        ("D", "s", 86400),
        ("M", "s", 2629800),
        ("Q", "s", 7889400),
        ("Y", "s", 31557600),
    ],
)
def test_seasonality_to_float(seasonality, freq, expected):
  """Make sure the seasonalities have the expected frequency."""
  assert spatiotemporal.seasonality_to_float(seasonality, freq) == expected


def test_seasonalities_to_array():
  periods = spatiotemporal.seasonalities_to_array(["D", "W", "M"], "h")
  np.testing.assert_allclose(periods, np.array([24, 168, 730.5]))


@pytest.mark.parametrize("p, h", [([], []), ([10, 15], [8, 6])])
def test_get_seasonality_periods_index(p, h):
  """Harmonics in discrete time are identical to inputs."""
  model = spatiotemporal.BayesianNeuralFieldMAP(
      freq="D",
      seasonality_periods=p,
      num_seasonal_harmonics=h,
      feature_cols=["t"],
      target_col="x",
      timetype="index",
  )
  assert np.all(model._get_seasonality_periods() == p)  # pylint: disable=protected-access
  assert np.all(model._get_num_seasonal_harmonics() == h)  # pylint: disable=protected-access


@pytest.mark.parametrize("p, h", [([], []), ([10, 12, .25], [.5, .5, .125])])
def test_get_seasonality_periods_float(p, h):
  """Harmonics in continuous time are min(.5, p/2)."""
  model = spatiotemporal.BayesianNeuralFieldMAP(
      seasonality_periods=p,
      feature_cols=["t"],
      target_col="x",
      timetype="float",
  )
  assert np.all(model._get_seasonality_periods() == p)  # pylint: disable=protected-access
  assert np.all(model._get_num_seasonal_harmonics() == h)  # pylint: disable=protected-access


def test_invalid_frequency():
  """timetype == 'index' requires a frequency."""
  model = spatiotemporal.BayesianNeuralFieldMAP(
      feature_cols=["t"],
      target_col="x",
      timetype="index",
  )
  with pytest.raises(ValueError):
    model._get_seasonality_periods()  # pylint: disable=protected-access

  # timetype == 'index' does not allow a frequency.
  model = spatiotemporal.BayesianNeuralFieldMAP(
      freq="M",
      feature_cols=["t"],
      target_col="x",
      timetype="float",
  )
  with pytest.raises(ValueError):
    model._get_seasonality_periods()  # pylint: disable=protected-access


def test_invalid_seasonality_period():
  # timetype == 'float' does not allow string periods.
  model = spatiotemporal.BayesianNeuralFieldMAP(
      seasonality_periods=["W"],
      feature_cols=["t"],
      target_col="x",
      timetype="float",
  )
  with pytest.raises(ValueError):
    model._get_seasonality_periods()  # pylint: disable=protected-access


def test_invalid_num_seasonal_harmonics():
  # timetype == 'float' does not allow num_seasonal_harmonics.
  model = spatiotemporal.BayesianNeuralFieldMAP(
      seasonality_periods=[1, 5],
      num_seasonal_harmonics=[0.5, 1],
      feature_cols=["t"],
      target_col="x",
      timetype="float",
  )
  with pytest.raises(ValueError):
    model._get_num_seasonal_harmonics()  # pylint: disable=protected-access
