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

"""Model code for Bayesian-neural-field time-series work."""

import enum
from typing import Tuple

import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp  # pylint: disable=g-importing-member

tfd = tfp.distributions


class LikelihoodDist(enum.Enum):
  NORMAL = 'NORMAL'
  NB = 'NB'
  ZINB = 'ZINB'


def make_seasonal_frequencies(
    seasonality_periods: np.ndarray, num_harmonics: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Return unique Fourier frequencies for given periods and harmonics."""
  seasonality_periods = np.array(seasonality_periods, dtype=np.float32)
  if np.any((num_harmonics > seasonality_periods / 2)):
    raise ValueError('Harmonic cannot exceed half seasonal period.')
  if seasonality_periods.shape != num_harmonics.shape:
    raise ValueError('Number of seasonal periods and harmonics must be equal.')
  if len(num_harmonics.shape) != 1:
    raise ValueError(
        'Arguments `num_harmonics` and `seasonality_periods` must be rank 1.'
    )
  if seasonality_periods.shape[0] == 0:
    return (np.zeros(0), np.zeros(0))
  harmonics = [np.arange(1, h + 1, dtype=np.float32) for h in num_harmonics]
  frequencies = np.concatenate(
      [h / p for (h, p) in zip(harmonics, seasonality_periods)]
  )
  _, idx = np.unique(frequencies, return_index=True)
  idx_sort = np.sort(idx)
  unique_frequencies = frequencies[idx_sort]
  unique_harmonics = np.concatenate(harmonics)[idx_sort]
  return (unique_frequencies, unique_harmonics)


def make_seasonal_features(
    x: jax.typing.ArrayLike,
    seasonality_periods: np.ndarray,
    num_harmonics: np.ndarray,
    rescale: bool = False,
) -> jnp.ndarray:
  """Returns a set of cos and sin features for each seasonality period."""
  x = jnp.reshape(x, (-1, 1))
  frequencies, harmonics = make_seasonal_frequencies(
      seasonality_periods, num_harmonics
  )
  y = 2 * jnp.pi * frequencies * x
  features = jnp.column_stack((jnp.cos(y), jnp.sin(y)))
  denominator = jnp.tile(harmonics, 2)
  return features / denominator if rescale else features


def make_fourier_features(
    x: jax.typing.ArrayLike, max_degree: int, rescale: bool = False
) -> jnp.ndarray:
  """Returns a set of sine and cosine basis functions."""
  x = jnp.reshape(x, (-1, 1))
  degrees = jnp.arange(max_degree)
  y = 2 * jnp.pi * 2**degrees * x
  features = jnp.column_stack((jnp.cos(y), jnp.sin(y)))
  denominator = jnp.tile(degrees + 1, 2)
  return features / denominator if rescale else features


prior_base_d = tfd.Logistic


def prior_model_fn(mlp_template):
  yield prior_base_d(0.0, 1.0)  # log_noise_scale
  yield prior_base_d(-1.5, 1.0)  # shape
  yield prior_base_d(0.0, 1.0)  # inflated_loc_probs

  leaves = jax.tree_util.tree_leaves(mlp_template)
  flat_mlp_params = []
  for p in leaves:
    weights = yield prior_base_d(0.0, jnp.ones_like(p))
    flat_mlp_params.append(weights)


def make_likelihood_model(
    params: jax.typing.ArrayLike,
    x: jax.Array,
    mlp: nn.Module,
    mlp_template: flax.core.frozen_dict.FrozenDict,
    distribution: str,
) -> tfd.Distribution:
  """Creates likelihood for the model.

  For a Gaussian likelihood, the likelihood is centered at the predictions and
  the scale is given by the noise scale.

  For a NegativeBinomial, we parametrize in terms of mean and shape
  following Salinas et al. "DeepAR: Probabilistic Forecasting with
  Autoregressive Recurrent Networks" https://arxiv.org/pdf/1704.04110.pdf
  (Section 3.1), but use a shared shape parameter, which is the inverse of the
  total count. The variance equals `mean + mean^2 * shape`.

  See also https://en.wikipedia.org/wiki/Negative_binomial_distribution, in
  particular the section on Alternative parameterizations, which defines:

  ```
  probs = mean / variance
  total_count = mean^2 / (variance - mean)
  ```

  so that `variance = mean + mean^2 / total_count`
  and thus `total_count = 1 / shape`.

  From this we get the parameters passed to `tfd.NegativeBinomial` below:

  ```
  total_count = 1 / shape
  probs = mean / (mean + mean^2 * shape)
        = 1 / (1 + mean * shape)
  ```

  For zero inflation we follow ideas from https://arxiv.org/pdf/2010.09647.pdf

  Args:
    params: Pytree of particles.
    x: Time indices for the training set.
    mlp: BayesianNeuralField1D model instance.
    mlp_template: MLP variables.
    distribution: Either 'NORMAL', 'NB' (Negative Binomial) or 'ZINB' (Zero
      Inflated Negative Binomial) are supported.

  Returns:
    A distribution with the likelihood for each sample in the batch.
  """
  if LikelihoodDist(distribution) == LikelihoodDist.NORMAL:
    log_noise_scale = params[0]
    treedef = jax.tree_util.tree_structure(mlp_template)
    mlp_params = jax.tree_util.tree_unflatten(treedef, params[3:])
    predictions = mlp.apply(mlp_params, x)
    return tfd.Independent(
        tfd.Normal(predictions, 0.01 + jnp.exp(log_noise_scale)), 1
    )

  elif LikelihoodDist(distribution) == LikelihoodDist.NB:
    treedef = jax.tree_util.tree_structure(mlp_template)
    mlp_params = jax.tree_util.tree_unflatten(treedef, params[3:])
    predictions = mlp.apply(mlp_params, x)
    mean = jax.nn.softplus(predictions)
    shape = jax.nn.softplus(params[1])

    negative_binomial = tfd.NegativeBinomial(
        total_count=1 / shape, logits=-jnp.log(shape) - jnp.log(mean)
    )
    return tfd.Independent(negative_binomial, 1)

  elif LikelihoodDist(distribution) == LikelihoodDist.ZINB:
    treedef = jax.tree_util.tree_structure(mlp_template)
    mlp_params = jax.tree_util.tree_unflatten(treedef, params[3:])
    predictions = mlp.apply(mlp_params, x)
    mean = jax.nn.softplus(predictions)
    shape = jax.nn.softplus(params[1])
    inflated_loc_probs = 1 / (1 + jnp.exp(-params[2]))

    zero_inflated_negative_binomial = tfd.ZeroInflatedNegativeBinomial(
        total_count=1 / shape,
        logits=-jnp.log(shape) - jnp.log(mean),
        inflated_loc_probs=inflated_loc_probs * jnp.ones(shape=mean.shape),
    )
    return tfd.Independent(zero_inflated_negative_binomial, 1)

  else:
    raise AssertionError('Unknown likelihood distribution:', distribution)


class BayesianNeuralField1D(nn.Module):
  """Linen Module implementing a 1D Bayesian neural field."""

  width: int
  depth: int
  input_scales: np.ndarray
  fourier_degrees: np.ndarray
  interactions: np.ndarray
  num_seasonal_harmonics: np.ndarray = flax.struct.field(
      default_factory=lambda: np.zeros((0,))
  )
  seasonality_periods: np.ndarray = flax.struct.field(
      default_factory=lambda: np.zeros((0,))
  )

  @nn.compact
  def __call__(self, x):
    init = nn.initializers.normal(1.0)

    if len(x.shape) == 1:
      x = x[..., jnp.newaxis]
    log_scale_adjustment = self.param(
        'log_scale_adjustment', init, x.shape[-1:]
    )
    scaled_x = x / (self.input_scales * jnp.exp(log_scale_adjustment))

    seasonal_features = make_seasonal_features(
        x[..., 0],
        self.seasonality_periods,
        self.num_seasonal_harmonics,
        rescale=True,
    )

    fourier_features = [
        make_fourier_features(scaled_x[..., i], degree, rescale=True)
        for i, degree in enumerate(self.fourier_degrees)
        if degree > 0
    ]

    interaction_features = jnp.prod(scaled_x[:, self.interactions], axis=-1)

    def make_layer_scale(name, shape=()):
      inv_sp_layer_scale = self.param(name, init, shape)
      return jax.nn.softplus(inv_sp_layer_scale)

    features = [
        scaled_x,
        *fourier_features,
        seasonal_features,
        interaction_features,
    ]
    features = [
        f * jax.nn.softplus(self.param(f'feature_inv_sp_scale{i}', init, ()))
        for i, f in enumerate(features) if f.size > 0
    ]
    h = jnp.concatenate(features, -1)

    activation_weight = jax.nn.sigmoid(
        self.param('logit_activation_weight', init, ())
    )

    def activation_fn(x):
      return activation_weight * nn.elu(x) + (1 - activation_weight) * nn.tanh(
          x
      )

    for layer_id in range(self.depth):
      layer = nn.Dense(self.width, kernel_init=init, bias_init=init)
      layer_scale = make_layer_scale(f'inv_sp_layer_scale{layer_id}')
      # Equivalent to scaling the variance of the weight prior by 1/num_inputs.
      h = h / jnp.sqrt(jnp.shape(h)[-1])
      h = activation_fn(layer_scale * layer(h))
    output_layer = nn.Dense(1, kernel_init=init, bias_init=init)
    output_scale = make_layer_scale('inv_sp_output_scale')
    # Equivalent to scaling the variance of the weight prior by 1/num_inputs.
    h = h / jnp.sqrt(h.shape[-1])
    return output_scale * output_layer(h)[..., 0]
