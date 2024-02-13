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

"""API for Bayesian Neural Field estimators."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp

from . import inference
from . import models

tfd = tfp.distributions


def seasonality_to_float(seasonality: str, freq: str) -> float:
  """Convert a valid pandas frequency string to a float, relative `freq`.

  Internally, this computes the number of `freq`s in a long period (to
  account for leap years), and the number of `seasonality`s, and returns the
  ratio.

  For example
  ```
  >>> seasonality_to_float('Y', 'D') == 365.25
  >>> seasonality_to_float('Y', 'W') == 52.25
  >>> seasonality_to_float('M', 'D') == 30.4375
  ```

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
  """Base class for BayesNF estimators.

  This class should not be initialized directly, but rather one of the three
  subclasses that implement different model learning procedures:

  - [BayesianNeuralFieldVI](BayesianNeuralFieldVI.md), for
    ensembles of surrogate posteriors from variational inference.

  - [BayesianNeuralFieldMAP](BayesianNeuralFieldMAP.md), for
    stochastic ensembles of maximum-a-posteriori estimates.

  - [BayesianNeuralFieldMLE](BayesianNeuralFieldMLE.md), for
    stochastic ensembles of maximum likelihood estimates.

  All three classes share the same `__init__` method described below.
  """

  _ensemble_dims: int
  _prior_weight: float = 1.0
  _scale_epochs_by_batch_size: bool = False

  def __init__(
      self,
      *,
      feature_cols: Sequence[str],
      target_col: str,
      seasonality_periods: Sequence[float | str] | None = None,
      num_seasonal_harmonics: Sequence[int] | None = None,
      fourier_degrees: Sequence[float] | None = None,
      interactions: Sequence[tuple[int, int]] | None = None,
      freq: str,
      timetype: str = 'index',
      depth: int = 2,
      width: int = 512,
      observation_model: str = 'NORMAL',
      standardize: Sequence[str] | None = None,
      ):
    """Shared initialization for subclasses of BayesianNeuralFieldEstimator.

    Args:
      feature_cols:
        Names of columns to use as features in the training
        data frame. The first entry denotes the name of the time variable,
        the remaining entries (if any) denote names of the spatial features.

      target_col:
        Name of the target column representing the spatial field.

      seasonality_periods:
        A list of numbers representing the seasonal frequencies of the data
        in the time domain. It is also possible to specify a string such as
        'W', 'D', etc. corresponding to a valid Pandas frequency: see the
        Pandas [Offset Aliases](
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
        for valid values.

      num_seasonal_harmonics:
        A list of seasonal harmonics, one for each entry in
        `seasonality_periods`. The number of seasonal harmonics (h) for a
        given seasonal period `p` must satisfy `h < p//2`.

      fourier_degrees:
        A list of integer degrees for the Fourier features of the inputs.
        If given, must have the same length as `feature_cols`.

      interactions:
        A list of tuples of column indexes for the first-order
        interactions. For example `[(0,1), (1,2)]` creates two interaction
        features

        - `feature_cols[0] * feature_cols[1]`
        - `feature_cols[1] * feature_cols[2]`

      freq:
        A frequency string for the sampling rate at which the data is
        collected. See the Pandas
        [Offset Aliases](
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
        for valid values.

      timetype:
        Must be specified as `index`. The general versions will be
        integrated pending https://github.com/google/bayesnf/issues/16.

      depth:
        The number of hidden layers in the BayesNF architecture.

      width:
        The number of hidden units in each layer.

      observation_model:
        The aleatoric noise model for the observed data. The options are
        `NORMAL` (Gaussian noise), `NB` (negative binomial noise), or `ZNB`
        (zero-inflated negative binomial noise).

      standardize:
        List of columns that should be standardized. It is highly
        recommended to standardize `feature_cols[1:]`. It is an error if
        `features_cols[0]` (the time variable) is in `standardize`.
    """
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
        'num_seasonal_harmonics':
            np.array(self.num_seasonal_harmonics)
            if self.num_seasonal_harmonics is not None
            else np.zeros(0),
        'seasonality_periods':
            seasonalities_to_array(self.seasonality_periods, self.freq)
            if self.seasonality_periods is not None
            else np.zeros(0),
        'width': self.width,
        'init_x': batch_shape,
        'fourier_degrees': self._get_fourier_degrees(batch_shape),
        'interactions': self._get_interactions(),
    }

  def predict(self, table, quantiles=(0.5,), approximate_quantiles=False):
    """Make predictions of the target column at new times.

    Args:
      table (pandas.DataFrame):
        Field locations at which to make new predictions. Same as `table` in
        [`fit`](), except that `self.target_col` need not be in `table`.

      quantiles (Sequence[float]):
        The list of quantiles to compute.

      approximate_quantiles (bool):
        If `False,` uses Chandrupatla root finding to compute quantiles.
        If `True`, uses a heuristic approximation of the quantiles.

    Returns:
      means (np.ndarray):
        The predicted means from each particle in the learned ensemble.
        The shape is `(num_devices, ensemble_size // num_devices, len(table))`
        and can be flattened to a 2D array using `np.row_stack(means)`.
        Related https://github.com/google/bayesnf/issues/17

      quantiles (List[np.ndarray]):
        A list of numpy arrays, one per requested quantile.
        The length of each array in the list is `len(table)`.

    """
    test_data = self.data_handler.get_test(table)
    return inference.predict_bnf(
        test_data,
        self.observation_model,
        params=self.params_,
        model_args=self._model_args(test_data.shape),
        quantiles=quantiles,
        ensemble_dims=self._ensemble_dims,
        approximate_quantiles=approximate_quantiles,
    )

  def fit(self, table, seed):
    """Run inference given a training data `table` and `seed`.

    Cannot be directly called on `BayesianNeuralFieldEstimator`.

    Args:
      table (pandas.DataFrame):
        A pandas DataFrame representing the
        training data. It has the following requirements:

        - The columns of `table` should contain all `self.feature_cols`
          and the `self.target_col`.

        - The type of the "time" column (i.e., `self.feature_cols[0]`)
          should be `datetime`. To ensure this requirement holds, see
          [`pandas.to_datetime`](
          https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html).
          The types of the remaining feature columns should be numeric.

      seed (jax.random.PRNGKey): The jax random key.
    """
    raise NotImplementedError('Should be implemented by subclass')

  def likelihood_model(self, table: pd.DataFrame) -> tfd.Distribution:
    """Access the predictive distribution over new field values in `table`.

    NOTE: Must be called after [`fit`]().

    Args:
      table (pandas.DataFrame):
        Field locations at which to make new predictions. Same as `table` in
        [`fit`](), except that `self.target_col` need not be in `table`.

    Returns:
      A probability distribution representing the predictive distribution
        over `self.target_col` at the new field values in `table`.
        See [tfp.distributions.Distribution](
        https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution)
        for the methods associated with this object.
    """
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
  """Fits models using stochastic ensembles of maximum-a-posteriori estimates.

  Implementation of
  [BayesianNeuralFieldEstimator](BayesianNeuralFieldEstimator.md).
  """

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
      ) -> BayesianNeuralFieldEstimator:
    """Run inference using stochastic MAP ensembles.

    Args:
      table (pandas.DataFrame):
        See documentation of
        [`table`][bayesnf.spatiotemporal.BayesianNeuralFieldEstimator.fit]
        in the base class.

      seed (jax.random.PRNGKey): The jax random key.

      ensemble_size (int): Number of particles in the ensemble. It currently
        an error if `ensemble_size < jax.device_count`, but will be fixed
        in https://github.com/google/bayesnf/issues/28.


      learning_rate (float): Learning rate for SGD.

      num_epochs (int): Number of full epochs through the training data.

      batch_size (None | int): Batch size for SGD. Default is `None`,
        meaning full-batch. Each epoch will perform `len(table) // batch_size`
        SGD updates.

      num_splits (int): Number of splits over the data to run training.
        Defaults to 1, meaning there are no splits.

    Returns:
      Instance of `self`.
    """
    if ensemble_size < jax.device_count():
      raise ValueError('ensemble_size cannot be smaller than device_count. '
                       'https://github.com/google/bayesnf/issues/28.')
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
  """Fits models using stochastic ensembles of maximum likelihood estimates.

  Implementation of
  [BayesianNeuralFieldEstimator](BayesianNeuralFieldEstimator.md).
  """

  _prior_weight = 0.0


class BayesianNeuralFieldVI(BayesianNeuralFieldEstimator):
  """Fits models using stochastic ensembles of surrogate posteriors from VI.

  Implementation of
  [BayesianNeuralFieldEstimator](BayesianNeuralFieldEstimator.md) using
  variational inference (VI).
  """

  _ensemble_dims = 3
  _scale_epochs_by_batch_size = True

  def fit(
      self,
      table,
      seed,
      ensemble_size=16,
      learning_rate=0.01,
      num_epochs=1_000,
      sample_size_posterior=30,
      sample_size_divergence=5,
      kl_weight=0.1,
      batch_size=None,
      ) -> BayesianNeuralFieldEstimator:
    """Run inference using stochastic variational inference ensembles.

    Args:
      table (pandas.DataFrame):
        See documentation of
        [`table`][bayesnf.spatiotemporal.BayesianNeuralFieldEstimator.fit]
        in the base class.

      seed (jax.random.PRNGKey): The jax random key.

      ensemble_size (int): Number of particles (i.e., surrogate posteriors)
        in the ensemble, **per device**. The available devices can be found
        via `jax.devices()`.

      learning_rate (float): Learning rate for SGD.

      num_epochs (int): Number of full epochs through the training data.

      sample_size_posterior (int): Number of samples of "posterior" model
        parameters draw from each surrogate posterior when making
        predictions.

      sample_size_divergence (int): number of Monte Carlo samples to use in
        estimating the variational divergence. Larger values may stabilize
        the optimization, but at higher cost per step in time and memory.
        See [`tfp.vi.fit_surrogate_posterior_stateless`](
        https://www.tensorflow.org/probability/api_docs/python/tfp/vi/fit_surrogate_posterior_stateless)
        for further details.

      kl_weight (float): Weighting of the KL divergence term in VI. The
        goal is to find a surrogate posterior `q(z)` that maximizes a
        version of the ELBO with the `KL(surrogate posterior || prior)`
        term scaled by `kl_weight`

            E_z~q [log p(x|z)] - kl_weight * KL(q || p)

        Reference
        > Weight Uncertainty in Neural Network
        > Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra.
        > Proceedings of the 32nd International Conference on Machine Learning.
        > PMLR 37:1613-1622, 2015.
        > <https://proceedings.mlr.press/v37/blundell15>

      batch_size (None | int): If specified, the log probability in each
        step of variational inference  is computed on a batch of this size.
        Default is `None`, meaning full-batch.

    Returns:
      Instance of self.
    """
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
        sample_size_posterior=sample_size_posterior,
        sample_size_divergence=sample_size_divergence,
        kl_weight=kl_weight,
        batch_size=batch_size,
    )
    return self
