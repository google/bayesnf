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

"""Model code for spatiotemporal Bayesian neural field."""
from collections.abc import Sequence

from bayesnf import inference
from bayesnf import models
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp


tfd = tfp.distributions


def seasonality_to_float(seasonality: str, freq: str) -> float:
  """Convert a valid pandas frequency string to a float, relative `freq`.

  Internally, this computes the number of `freq`s in a long period (to
  account for leap years), and the number of `seasonality`s, and returns the
  ratio.

  For example
    seasonality_to_float('Y', 'D') == 365.25

    seasonality_to_float('Y', 'W') == 52.25

    seasonality_to_float('M', 'D') == 30.4375

  Args:
    seasonality: A valid pandas frequency.
    freq: A valid pandas frequency. Should be shorter than `seasonality`.

  Returns:
    A float of how many `seasonality` periods are in a `freq`, on average.
  """
  four_years = pd.date_range('2020-01-01', periods=5, freq='Y')
  y = four_years.to_period(seasonality)
  num_seasonality = (y[-1] - y[0]).n

  x = pd.date_range(y[0].start_time, y[-1].start_time).to_period(freq)
  num_freq = (x[-1] - x[0]).n

  return num_freq / num_seasonality


def seasonalities_to_array(
    seasonalities: Sequence[float | str], freq: str
) -> np.ndarray:
  """Convert a list of floats or strings to durations relative to a frequency.

  Args:
    seasonalities: A list of floats or strings representing durations relative
      to `freq`. For example, if `freq` is 'D', either 365 or 'Y' would
      represent a year. If `freq` is 'M', then either 12 or 'Y' represents a
      year.
    freq: Frequency of the data.

  Raises:
    TypeError: If the seasonality is less than or equal to 1.

  Returns:
    Array of floats greater than 1, representing seasonalities.
  """
  ret = []
  for seasonality in seasonalities:
    if isinstance(seasonality, str):
      seasonality_float = seasonality_to_float(seasonality, freq)
      if seasonality_float < 1:
        raise TypeError(f'{seasonality=} should represent a time '
                        f'span greater than {freq=}, but {seasonality} '
                        f'is {seasonality_float:.2f} of a {freq}')

    else:
      seasonality_float = seasonality
      if seasonality_float < 1:
        raise TypeError(f'{seasonality_float=} should be larger than 1.')
    ret.append(seasonality_float)
  return np.array(ret)


def _convert_datetime_col(table, time_column, timetype, freq, time_min=None):
  if timetype == 'index':
    first_date = pd.to_datetime('2020-01-01').to_period(freq)
    table[time_column] = table[time_column].dt.to_period(freq)
    table[time_column] = (table[time_column] - first_date).apply(lambda x: x.n)
  if time_min is None:
    time_min = table[time_column].min()
  table[time_column] = table[time_column] - time_min
  return table, time_min


class SpatiotemporalDataHandler:
  """Base class for preparing spatiotemporal data."""

  def __init__(
      self,
      feature_cols: Sequence[str],
      target_col: str,
      timetype: str,
      freq: str,
      standardize: Sequence[str] | None = None,
  ):
    self.feature_cols = feature_cols
    self.target_col = target_col
    self.timetype = timetype
    self.freq = freq
    self.standardize = standardize
    self.mu_ = None
    self.std_ = None
    self.time_min_ = None
    self.time_scale_ = None

  @property
  def _time_idx(self) -> int:
    return 0

  @property
  def _time_column(self) -> str:
    return self.feature_cols[self._time_idx]

  def get_target(self, table: pd.DataFrame) -> np.ndarray:
    table = self._maybe_filter_target_nans(table)
    return table[self.target_col].values

  def _maybe_filter_target_nans(self, table: pd.DataFrame) -> pd.DataFrame:
    if self.target_col in table.columns:
      return table[table[self.target_col].notna()]
    return table

  def copy_and_filter_table(self, table: pd.DataFrame) -> pd.DataFrame:
    return self._maybe_filter_target_nans(table.copy())

  def get_train(self, table: pd.DataFrame) -> np.ndarray:
    """Fetch training data."""
    table = self.copy_and_filter_table(table)
    self.mu_ = np.zeros(len(self.feature_cols))
    self.std_ = np.ones(len(self.feature_cols))

    table, self.time_min_ = _convert_datetime_col(
        table, self._time_column, self.timetype, self.freq, None)
    features = table[self.feature_cols].values
    self.time_scale_ = features[:, self._time_idx].max()

    if self.standardize:
      if self._time_column in self.standardize:
        raise TypeError('Do not standardize the time column!')
      idx = [self.feature_cols.index(f) for f in self.standardize]
      self.mu_[idx] = np.mean(features[:, idx].astype(float), axis=0)
      self.std_[idx] = np.std(features[:, idx].astype(float), axis=0)
      features = (features - self.mu_) / self.std_

    return features

  def get_test(self, table: pd.DataFrame) -> np.ndarray:
    """Fetch testing data. Call this after `get_train`."""
    table = self.copy_and_filter_table(table)
    table, _ = _convert_datetime_col(
        table, self._time_column, self.timetype, self.freq, self.time_min_)

    features = table[self.feature_cols].values

    if self.standardize:
      features = (features - self.mu_) / self.std_

    return features

  def get_input_scales(self) -> np.ndarray:
    input_scales = np.ones(len(self.feature_cols))
    input_scales[self._time_idx] = self.time_scale_
    return input_scales


class BayesianNeuralFieldEstimator:
  """Base class for BNF estimator."""

  _ensemble_dims: int
  _prior_weight: float = 1.0
  _scale_epochs_by_batch_size: bool = False

  def __init__(
      self,
      num_seasonal_harmonics: Sequence[int],
      seasonality_periods: Sequence[float | str],
      feature_cols: Sequence[str],
      target_col: str,
      timetype: str,
      freq: str,
      fourier_degrees: Sequence[float] | None = None,
      observation_model: str = 'NORMAL',
      depth: int = 2,
      width: int = 512,
      standardize: Sequence[str] | None = None,
      interactions: Sequence[tuple[int, int]] | None = None,
  ):
    self.num_seasonal_harmonics = num_seasonal_harmonics
    self.seasonality_periods = seasonality_periods
    self.observation_model = observation_model
    self.depth = depth
    self.width = width
    self.feature_cols = feature_cols
    self.target_col = target_col
    self.timetype = timetype
    self.freq = freq
    self.fourier_degrees = fourier_degrees
    self.standardize = standardize
    self.interactions = interactions

    self.losses_ = None
    self.params_ = None
    self.data_handler = SpatiotemporalDataHandler(
        self.feature_cols,
        self.target_col,
        self.timetype,
        self.freq,
        standardize=self.standardize)

  def _get_fourier_degrees(self, batch_shape: tuple[int, ...]) -> np.ndarray:
    """Set default fourier degrees, or verify shape is correct."""
    if self.fourier_degrees is None:
      fourier_degrees = np.full(batch_shape[-1], 5, dtype=int)
    else:
      fourier_degrees = np.atleast_1d(self.fourier_degrees).astype(int)
      if fourier_degrees.shape[-1] != batch_shape[-1]:
        raise ValueError(
            'The length of fourier_degrees ({}) must match the '
            'input dimension dimension ({}).'.format(
                fourier_degrees.shape[-1], batch_shape[-1]
            )
        )
    return fourier_degrees

  def _get_interactions(self) -> np.ndarray:
    """Set default fourier degrees, or verify shape is correct."""
    if self.interactions is None:
      interactions = np.zeros((0, 2), dtype=int)
    else:
      interactions = np.array(self.interactions).astype(int)
      if np.ndim(interactions) != 2 or interactions.shape[-1] != 2:
        raise ValueError(
            'The argument for `interactions` should be a 2-d array of integers '
            'of shape (N, 2), indicating the column indices to interact (the '
            f' passed shape was {interactions.shape})')
    return interactions

  def _model_args(self, batch_shape):
    return {
        'depth': self.depth,
        'input_scales': self.data_handler.get_input_scales(),
        'num_seasonal_harmonics': np.array(self.num_seasonal_harmonics),
        'seasonality_periods': seasonalities_to_array(
            self.seasonality_periods, self.freq),
        'width': self.width,
        'init_x': batch_shape,
        'fourier_degrees': self._get_fourier_degrees(batch_shape),
        'interactions': self._get_interactions(),
    }

  def predict(self, table, quantiles=(0.5,)):
    test_data = self.data_handler.get_test(table)
    return inference.predict_bnf(
        test_data,
        self.observation_model,
        params=self.params_,
        model_args=self._model_args(test_data.shape),
        quantiles=quantiles,
        ensemble_dims=self._ensemble_dims,
        approximate_quantiles=False,
    )

  def fit(self, table, seed):
    raise NotImplementedError('Should be implemented by subclass')

  def likelihood_model(self, table: pd.DataFrame) -> tfd.Distribution:
    """Access the likelihood distribution after calling `.fit`."""
    test_data = self.data_handler.get_test(table)
    mlp, mlp_template = inference.make_model(
        **self._model_args(test_data.shape))
    for _ in range(self._ensemble_dims - 1):
      mlp.apply = jax.vmap(mlp.apply, in_axes=(0, None))
    mlp.apply = jax.pmap(mlp.apply, in_axes=(0, None))

    # This allows the likelihood to broadcast correctly with the batch of
    # predictions.
    params = self.params_._replace(**{  # pytype: disable=attribute-error
        self.params_._fields[i]: self.params_[i][..., jnp.newaxis]  # pytype: disable=unsupported-operands,attribute-error
        for i in range(3)})

    return models.make_likelihood_model(
        params,
        jnp.array(test_data),
        mlp,
        mlp_template,
        self.observation_model)


class BayesianNeuralFieldMAP(BayesianNeuralFieldEstimator):
  """Fit BNF using MAP estimation."""

  _ensemble_dims = 2

  def fit(
      self,
      table,
      seed,
      ensemble_size=16,
      learning_rate=0.005,
      num_epochs=5_000,
      batch_size=None,
      num_splits=1,
  ):
    train_data = self.data_handler.get_train(table)
    train_target = self.data_handler.get_target(table)
    if batch_size is None:
      batch_size = train_data.shape[0]
    if self._scale_epochs_by_batch_size:
      num_epochs = num_epochs * (train_data.shape[0] // batch_size)
    model_args = self._model_args((batch_size, train_data.shape[-1]))
    self.params_, self.losses_ = inference.fit_map(
        train_data,
        train_target,
        seed=seed,
        observation_model=self.observation_model,
        model_args=model_args,
        num_particles=ensemble_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        prior_weight=self._prior_weight,
        batch_size=batch_size,
        num_splits=num_splits)
    return self


class BayesianNeuralFieldMLE(BayesianNeuralFieldMAP):
  """Fit BNF using MLE estimation."""

  _prior_weight = 0.0

  def predict(self, table, quantiles=(0.5,)):
    test_data = self.data_handler.get_test(table)
    return inference.predict_bnf(
        test_data,
        self.observation_model,
        params=self.params_,
        model_args=self._model_args(test_data.shape),
        quantiles=quantiles,
        ensemble_dims=self._ensemble_dims,
        approximate_quantiles=True,
    )


class BayesianNeuralFieldVI(BayesianNeuralFieldEstimator):
  """Fit BNF using VI estimation."""

  _ensemble_dims = 3
  _scale_epochs_by_batch_size = True

  def fit(
      self,
      table,
      seed,
      ensemble_size=16,
      learning_rate=0.01,
      num_epochs=1_000,
      sample_size=5,
      kl_weight=0.1,
      batch_size=None,
  ):
    train_data = self.data_handler.get_train(table)
    train_target = self.data_handler.get_target(table)
    if batch_size is None:
      batch_size = train_data.shape[0]
    if self._scale_epochs_by_batch_size:
      num_epochs = num_epochs * (train_data.shape[0] // batch_size)
    model_args = self._model_args((batch_size, train_data.shape[-1]))
    _, self.losses_, self.params_ = inference.fit_vi(
        train_data,
        train_target,
        seed=seed,
        observation_model=self.observation_model,
        model_args=model_args,
        ensemble_size=ensemble_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        sample_size=sample_size,
        kl_weight=kl_weight,
        batch_size=batch_size,
    )
    return self
