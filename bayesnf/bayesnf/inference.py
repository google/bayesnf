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

"""Code to run inference using spatiotemporal BNF."""

import functools
from typing import Any, Callable

from bayesnf import models
import flax
import jax
from jax import numpy as jnp
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp  # pylint: disable=g-importing-member


tfd = tfp.distributions
ArrayT = jax.Array | np.ndarray


def permute_dataset(
    features: ArrayT, target: ArrayT, seed: jax.Array
) -> tuple[ArrayT, ArrayT]:
  permutation = jax.random.permutation(seed, jnp.arange(target.shape[0]))
  return features[permutation], target[permutation]


@functools.partial(jax.jit, static_argnames=('axis',))
def _normal_quantile_via_root(means, scales, q, axis=(0, 1)):
  n = tfd.Normal(means, scales)
  res = tfp.math.find_root_chandrupatla(
      lambda x: n.cdf(x).mean(axis) - q,
      low=jnp.amin(means) - 5 * jnp.amax(scales),
      high=jnp.amax(means) + 5 * jnp.amax(scales),
      value_tolerance=1e-5,
      max_iterations=60,
  )
  return res.estimated_root


@functools.partial(jax.jit, static_argnames=('axis',))
def _approximate_normal_quantile(
    means: jax.Array, scales: jax.Array, q: float, axis=(0, 1)
) -> jax.Array:
  """Fast approximate quantile for a mixture of gaussians.

  Compute the mean and standard deviation of the mixture distribution,
  then pretend the mixture distribution is a Normal distribution and
  compute Normal quantiles, as in [1].

  [1] Lakshminarayanan, Pritzel, and Blundell. "Simple and scalable
      predictive uncertainty estimation using deep ensembles." NeurIPS 2017.

  Args:
    means: Array of normal means
    scales: Array of normal scales. Should broadcast with `means`.
    q: Quantile to compute in (0, 1).
    axis: Batch axis to reduce over.

  Returns:
    Array of approximate `q`th quantiles, which should have
    shape `means.shape[-1]`.
  """
  mixture_mean = means.mean(axis)
  mixture_scale = jnp.sqrt(
      (jnp.square(scales) + jnp.square(means)).mean(axis)
      - jnp.square(mixture_mean)
  )
  n = tfd.Normal(mixture_mean, mixture_scale)
  return n.quantile(q)


def _get_percentile_normal(
    means, scales, quantiles, axis=(0, 1), approximate=False
) -> list[jax.Array]:
  """Compute the normal percentiles for a mixture."""
  if approximate:
    quantile_fn = _approximate_normal_quantile
  else:
    quantile_fn = _normal_quantile_via_root
  forecast_quantiles = []
  for q in quantiles:
    forecast_quantiles.append(
        quantile_fn(means, scales[..., jnp.newaxis], q, axis)
    )
  return forecast_quantiles


def _make_forecast_inner(model_args, distribution):
  """Construct inner forecast function for MAP and VI."""

  def forecast_inner(params, x_subset):
    likelihood = models.make_likelihood_model(
        params, x_subset, *make_model(**model_args), distribution
    )
    if distribution == models.LikelihoodDist.NORMAL:
      return (likelihood.distribution.loc, likelihood.distribution.scale)
    elif distribution == models.LikelihoodDist.NB:
      return (
          likelihood.distribution.total_count,
          likelihood.distribution.logits,
      )
    elif distribution == models.LikelihoodDist.ZINB:
      return (
          likelihood.distribution.total_count,
          likelihood.distribution.logits,
          likelihood.distribution.inflated_loc_probs,
      )
    else:
      raise TypeError('Distribution must be one of NORMAL, NB, or ZINB.')

  return forecast_inner


def forecast_parameters_batched(
    features: jax.Array,
    params: PyTree,
    distribution: models.LikelihoodDist,
    forecast_inner: Callable[[PyTree, jax.Array], PyTree],
    batchsize: int = 1024,
) -> tuple[jax.Array, ...]:
  """Computes parameters of the likelihood distribution.

  To handle large datasets, this operation is batched.

  Args:
    features: Input data to compute parameters over.
    params: Pytree of model parameters.
    distribution: Name of likelihood distribution.
    forecast_inner: Inner function used for forecasting. Use
      `_make_forecast_inner` to construct.
    batchsize: Batch size to use for splitting up the computation.

  Returns:
    A tuple of parameters, depending on the likelihood distribution.
    Normal likelihood: `loc` of shape BATCH_SHAPE+OBS_SHAPE and `scale` of
      shape BATCH_SHAPE
    Negative Binomial: `total_count` of shape BATCH_SHAPE and `logits` of
      shape BATCH_SHAPE + OBS_SHAPE
    Zero Inflated Negative Binomial: `total_count` of shape BATCH_SHAPE,
      `logits` of shape BATCH_SHAPE + OBS_SHAPE, and  `inflated_loc_probs` of
      shape BATCH_SHAPE.
    Note that `loc` and `logits` have shape BATCH_SHAPE + OBS_SHAPE. Other
    parameters are shared across time and only have BATCH_SHAPE.
  """
  forecast_params_slices = [[], [], []]

  data_size = features.shape[0]
  num_batches = data_size // batchsize
  for i in range(num_batches + 1):
    if i == num_batches:
      # last batch can be a partial batch
      batch_slice = slice(i * batchsize, None)
      if not batch_slice:
        # empty batch
        continue
    else:
      batch_slice = slice(i * batchsize, (i + 1) * batchsize)
    forecast_params = forecast_inner(params, features[batch_slice])
    for i, fc_param in enumerate(forecast_params):
      # This loop can be over 2 params for NORMAL and NB or 3 params for ZINB:
      # (count, logit) or (loc, scale), or (count, logit, inflated_loc_scale)
      forecast_params_slices[i].append(fc_param)

  # logit and loc are per sample so we concatenate across batches:
  if distribution == models.LikelihoodDist('NORMAL'):
    loc = jnp.concatenate(forecast_params_slices.pop(0), axis=-1)
    forecast_params = [k[0] for k in forecast_params_slices if k]
    forecast_params.insert(0, loc)
  elif distribution == models.LikelihoodDist('NB'):
    logit = jnp.concatenate(forecast_params_slices.pop(1), axis=-1)
    forecast_params = [k[0] for k in forecast_params_slices if k]
    forecast_params.insert(1, logit)
  elif distribution == models.LikelihoodDist('ZINB'):
    count = forecast_params_slices[0][0]
    logit = jnp.concatenate(forecast_params_slices[1], axis=-1)
    zero_mass = jnp.concatenate(forecast_params_slices[2], axis=-1)
    forecast_params = count, logit, zero_mass
  else:
    raise TypeError('Distribution must be NORMAL, NB, or ZINB.')

  return tuple(forecast_params)


def make_vi_init(prior_d: tfd.Distribution):
  """Given a JDS prior, construct a surrogate posterior init function."""
  xs = prior_d.sample(seed=jax.random.PRNGKey(0))

  def _fn():
    for i, x in enumerate(xs):
      # Surrogate mean.
      if len(x.shape) != 2:
        yield tfd.Deterministic(
            jnp.zeros_like(x),
            name=f'zero_initial_mean_for_bias_or_transformed_scale_{i}',
        )
      else:
        yield tfd.TruncatedNormal(
            0.0,
            jnp.ones_like(x),
            low=-2,
            high=2,
            name=f'initial_weight_matrix_{i}',
        )

      yield tfd.Deterministic(
          tfp.math.softplus_inverse(0.3) * jnp.ones_like(x),
          name=f'initial_inv_softplus_surrogate_scale_{i}',
      )

  return tfd.JointDistributionCoroutine(
      _fn, use_vectorized_map=True, batch_ndims=0
  ).sample


def make_model(
    width: int,
    depth: int,
    input_scales: np.ndarray,
    num_seasonal_harmonics: np.ndarray,
    seasonality_periods: np.ndarray,
    init_x: tuple[int, ...],
    fourier_degrees: np.ndarray,
    interactions: np.ndarray,
) -> tuple[models.BayesianNeuralField1D, flax.core.scope.FrozenVariableDict]:
  """Instantiate and initialize BayesianNeuralField1D model."""
  mlp = models.BayesianNeuralField1D(
      width=width,
      depth=depth,
      input_scales=input_scales,
      fourier_degrees=fourier_degrees,
      interactions=interactions,
      num_seasonal_harmonics=num_seasonal_harmonics,
      seasonality_periods=seasonality_periods,
  )
  mlp_template = mlp.init(
      jax.random.PRNGKey(0), jnp.zeros(init_x, dtype=jnp.float32)
  )

  return mlp, mlp_template


def make_prior(**kwargs: dict[str, Any]) -> tfd.JointDistributionCoroutine:
  kwargs.pop('likelihood_distribution', None)
  prior_d = tfd.JointDistributionCoroutine(
      functools.partial(models.prior_model_fn, make_model(**kwargs)[1]),
      use_vectorized_map=True,
      batch_ndims=0,
  )
  return prior_d


def _build_observation_distribution(distribution, forecast_params):
  """Returns (zero inflated) Negative Binomial distibution given parameters.

  Args:
    distribution: Indicates whether a zero-inflated models.LikelihoodDist.ZINB
      or models.LikelihoodDist.NB disribution should be returned.
    forecast_params: Tuple of total_count, logits, and optionally
      maybe_zero_mass
  """
  total_count, logits, *maybe_zero_mass = forecast_params
  if distribution == models.LikelihoodDist.NB:
    return tfd.NegativeBinomial(
        total_count=total_count[..., jnp.newaxis], logits=logits
    )

  elif distribution == models.LikelihoodDist.ZINB:
    inflated_loc_probs = maybe_zero_mass[0]

    return tfd.ZeroInflatedNegativeBinomial(
        total_count=total_count[..., jnp.newaxis],
        logits=logits,
        inflated_loc_probs=inflated_loc_probs,
    )
  else:
    raise ValueError(f'Unknown distribution: {distribution}')


@functools.partial(jax.jit, static_argnames=('ensemble_axes',))
def _get_nb_quantiles_root(
    dist: tfd.Distribution, q: float, ensemble_axes: tuple[int, ...] = (0, 1, 2)
) -> jax.Array:
  """Returns (zero inflated) Negative Binomial quantiles via root-finding.

  Quantiles are computed using root-finding. We post-process the result from
  `_nb_quantile_via_root` by taking the ceiling. Since the CDF is already
  above zero for an input of zero, we filter these out and manually set the
  values to zero. The upper limit for the root finding algorithm is based on
  the Chebyshev bound (http://en.wikipedia.org/wiki/Chebyshev%27s_inequality).

  Note that this is intended to be used with VI and MAP, and not AIS, where
  particles weights need to be incorporated into the average.

  Args:
    dist: Distribution, assumed to to be (zero inflated) Negative Binomial.
    q: Quantile to estimate.
    ensemble_axes: Int or tuple of ints that index ensemble dimensions. Default
      value is (0, 1, 2).
  """
  res = tfp.math.find_root_chandrupatla(
      lambda x: dist.cdf(x).mean(axis=ensemble_axes) - q,
      low=0.0,
      high=(
          jnp.amax(dist.mean())
          + 1.1 * jax.lax.rsqrt(1 - q) * jnp.amax(dist.stddev())
      ),
      value_tolerance=1e-5,
      max_iterations=60,
  )
  return jnp.ceil(
      jnp.where(
          dist.prob(0).mean(axis=ensemble_axes) > q, 0, res.estimated_root
      )
  )


def fit_vi(
    features: ArrayT,
    target: ArrayT,
    seed: jax.Array,
    observation_model: str,
    model_args: dict[str, Any],
    ensemble_size: int,
    learning_rate: float,
    num_epochs: int,
    sample_size: int,
    kl_weight: float,
    batch_size: int | None = None,
) -> tuple[
    tfp.distributions.JointDistributionSequential, jax.Array, jax.Array
]:
  """Fit BNF using an ensemble VI."""
  distribution = models.LikelihoodDist(observation_model)

  def _neg_energy_fn(params, x, y):
    return models.make_likelihood_model(
        params, x, *make_model(**model_args), distribution
    ).log_prob(y)

  return ensemble_vi(
      features,
      target,
      _neg_energy_fn,
      prior_d=make_prior(**model_args),
      ensemble_size=ensemble_size // jax.device_count(),
      learning_rate=learning_rate,
      num_epochs=num_epochs,
      seed=seed,
      sample_size=sample_size,
      kl_weight=kl_weight,
      batch_size=batch_size,
  )


def fit_map(
    features: ArrayT,
    target: ArrayT,
    seed: jax.Array,
    observation_model: str,
    model_args: dict[str, Any],
    num_particles: int,
    learning_rate: float,
    num_epochs: int,
    prior_weight: float = 1.0,
    batch_size: int | None = None,
    num_splits: int = 1,
) -> tuple[PyTree, np.ndarray]:
  """Fit a BNF using provided features and y data."""
  distribution = models.LikelihoodDist(observation_model)

  def _neg_energy_fn(params, x, y):
    return models.make_likelihood_model(
        params, x, *make_model(**model_args), distribution
    ).log_prob(y)

  target_scale = np.nanstd(target)

  def _make_init_fn(prior_d):
    xs = prior_d.sample(seed=jax.random.PRNGKey(0))

    def _fn():
      for i, x in enumerate(xs):
        if i == 0:
          # Initialize the noise scale (for the Normal observation distribution)
          # to be in the ballpark of the standard deviation of the observations.
          yield tfd.Deterministic(
              jnp.ones_like(x) * jnp.log(target_scale / 2.0),
              name='zero_initial_log_noise_scale',
          )
        elif len(x.shape) != 2:
          yield tfd.Deterministic(
              jnp.zeros_like(x),
              name=f'zero_initial_mean_for_bias_or_transformed_scale_{i}',
          )
        else:
          yield tfd.TruncatedNormal(
              0.0,
              jnp.ones_like(x),
              low=-2,
              high=2,
              name=f'initial_weight_matrix_{i}',
          )

    return lambda seed: tfd.JointDistributionCoroutine(
        _fn, use_vectorized_map=True, batch_ndims=0
    ).sample(seed=seed)

  prior = make_prior(**model_args)
  params = []
  losses = []
  for i in range(num_splits):
    if num_splits > 1:
      seed_i = jax.random.fold_in(seed, i)
    else:
      # Avoid changing the seed when num_splits is 1, for better comparisons
      # with previous experiments.
      seed_i = seed
    params_i, losses_i = ensemble_map(
        features,
        target,
        _neg_energy_fn,
        prior_d=prior,
        init_fn=_make_init_fn(prior),
        ensemble_size=((num_particles // num_splits) // jax.device_count()),
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        seed=seed_i,
        batch_size=batch_size,
        prior_weight=prior_weight,
    )
    params.append(jax.tree_util.tree_map(np.array, params_i))
    losses.append(np.array(losses_i))
  params = jax.tree_util.tree_map(
      lambda *ts: np.concatenate(ts, axis=1), *params
  )
  losses = np.concatenate(losses, axis=1)
  return params, losses


def predict_bnf(
    features: ArrayT,
    observation_model: str,
    params: PyTree,
    model_args: dict[str, Any],
    quantiles: jax.Array,
    ensemble_dims: int = 2,
    approximate_quantiles: bool = False,
) -> tuple[jax.Array, list[jax.Array]]:
  """Predict new data from an existing BNF fit."""
  distribution = models.LikelihoodDist(observation_model)
  assert ensemble_dims >= 1

  forecast_inner = _make_forecast_inner(model_args, distribution)
  for _ in range(ensemble_dims - 1):
    forecast_inner = jax.vmap(forecast_inner, in_axes=(0, None))
  forecast_inner = jax.pmap(forecast_inner, in_axes=(0, None))
  axis = tuple(range(ensemble_dims))

  forecast_params = forecast_parameters_batched(
      features, params, distribution, forecast_inner
  )
  if distribution == models.LikelihoodDist.NORMAL:
    (means, scales) = forecast_params
    forecast_means = means
    forecast_quantiles = _get_percentile_normal(
        forecast_means,
        scales,
        quantiles,
        axis=axis,
        approximate=approximate_quantiles,
    )

  elif (
      distribution == models.LikelihoodDist.NB
      or distribution == models.LikelihoodDist.ZINB
  ):
    obs_d = _build_observation_distribution(distribution, forecast_params)
    forecast_means = obs_d.mean()
    forecast_quantiles = jax.vmap(
        lambda q: _get_nb_quantiles_root(obs_d, q, ensemble_axes=axis)
    )(jnp.array(quantiles))

  else:
    raise ValueError(f'Unknown distribution: {distribution}')

  return forecast_means, forecast_quantiles


def ensemble_map(
    features: ArrayT,
    target: ArrayT,
    neg_energy_fn: Callable[[PyTree, jax.Array, jax.Array], float],
    prior_d: tfd.Distribution,
    init_fn: Callable[[jax.Array], PyTree],
    ensemble_size: int,
    learning_rate: float,
    num_epochs: int,
    seed: jax.Array,
    batch_size: int | None = None,
    prior_weight: float = 1.0,
) -> tuple[PyTree, jax.Array]:
  """Fit an ensemble of MAP estimates.

  Args:
    features: Training features.
    target: Training targets.
    neg_energy_fn: The callable log-likelihood function.
    prior_d: `tfp.distributions.JointDistributionSequential` defining the prior.
    init_fn: Callable that, given a jax.random.PRNGKey, returns initial
      parameters for optimization.
    ensemble_size: `int` number of estimates to fit **per device**.
    learning_rate: `float` learning rate for optimization.
    num_epochs: `int` number of epochs of optimization to run.
    seed: `jax.random.PRNGKey` random seed.
    batch_size: `int` batch size to use during optimization.  The dataset is
      split into batches of this size along dimension 0.  If the batch size does
      not evenly divide the size of the dataset, the final ragged batch is
      skipped each epoch.  Default: None, indicating that the entire dataset
      should be used as a single batch.
    prior_weight: A positive `float`.  When computing the loss, the prior
      log-prob is multiplied by `prior_weight`.

  Returns:
    params: Inferred parameters.  The leftmost dimensions will be
      `(NUM_DEVICES, ensemble_size)`.
    losses: Array of shape (NUM_DEVICES, ensemble_size, num_epochs) -- the
      losses from each step (epoch) of optimization.
  """
  features = jnp.array(features)
  target = jnp.array(target)
  if batch_size is None:
    batch_size = target.shape[0]

  def _target_log_prob_fn(params, x_batch, y_batch):
    # given one pytree of model params and a batch of data,
    # return the loss on the batch.
    if prior_weight == 0.0:
      return -(
          neg_energy_fn(params, x_batch, y_batch)
          * (target.shape[0] / batch_size))
    else:
      return -(
          neg_energy_fn(params, x_batch, y_batch)
          * (target.shape[0] / batch_size)
          + prior_d.log_prob(params) * prior_weight)

  init_seed, opt_seed = jax.random.split(seed, 2)
  num_devices = jax.device_count()
  init_params = jax.vmap(jax.vmap(init_fn))(
      jax.random.split(init_seed, (num_devices, ensemble_size))
  )

  @jax.pmap
  @jax.vmap
  def _run(init_params, seed):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_params)

    def _reshape_to_batches(t):
      t = jax.tree_util.tree_map(
          lambda v: v[: (v.shape[0] // batch_size) * batch_size], t
      )
      return jax.tree_util.tree_map(
          lambda v: v.reshape((-1, batch_size) + v.shape[1:]), t
      )

    def _one_epoch(carry, _):
      params, opt_state, seed = carry
      seed, permute_seed = jax.random.split(seed, 2)
      if batch_size < target.shape[0]:
        x, y = permute_dataset(features, target, permute_seed)
      else:
        x, y = features, target

      def _one_step(carry, batch):
        params, opt_state = carry
        batch_x, batch_y = batch
        loss, grads = jax.value_and_grad(_target_log_prob_fn)(
            params, batch_x, batch_y
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

      (params, opt_state), losses = jax.lax.scan(
          _one_step,
          (params, opt_state),
          (_reshape_to_batches(x), _reshape_to_batches(y)),
      )
      return (params, opt_state, seed), losses.mean()

    (params, _, _), losses = jax.lax.scan(
        _one_epoch, (init_params, opt_state, seed), None, length=num_epochs
    )
    return params, losses

  return _run(
      init_params, jax.random.split(opt_seed, (num_devices, ensemble_size))
  )


def ensemble_vi(
    features: ArrayT,
    target: ArrayT,
    neg_energy_fn: Callable[[PyTree, jax.Array, jax.Array], float],
    prior_d: tfd.Distribution,
    ensemble_size: int,
    learning_rate: float,
    num_epochs: int,
    seed: jax.Array,
    sample_size: int = 10,
    num_samples: int = 30,
    kl_weight: float = 1.0,
    batch_size: int | None = None,
) -> tuple[
    tfp.distributions.JointDistributionSequential, jax.Array, jax.Array
]:
  """Fit an ensemble of surrogate posteriors.

  NOTE: For now, this function uses full-batch Variational Inference.

  Args:
    features: features for fitting.
    target: An array of target values.
    neg_energy_fn: The callable log-likelihood function.
    prior_d: `tfp.distributions.JointDistributionSequential` defining the prior.
    ensemble_size: `int` number of surrogate posteriors to fit **per device**.
    learning_rate: `float` learning rate for optimization.
    num_epochs: `int` number of epochs of optimization to run.
    seed: `jax.random.PRNGKey` random seed.
    sample_size: `int` number of Monte Carlo samples to use in estimating the
      variational divergence.
    num_samples: `int` number of posterior samples to return.
    kl_weight: A positive `float`.  During inference, we attempt to find a
      surrogate posterior `q(z)` that maximizes a version of the ELBO with the
      `KL(surrogate posterior || prior)` term scaled by `kl_weight` -- i.e.
      `E_z~q [log p(x|z)] - kl_weight * KL(q || p)`.
    batch_size: `int` batch size.  If specified, the log-prob in each step of VI
      is computed on a batch of this size, instead of on the whole dataset.
      Defaults to `None`.

  Returns:
    surrogate_posterior: A distribution over the (approximate) posterior.
      for the learned mean-field surrogate posteriors.  The leftmost
      dimensions will be (NUM_DEVICES, ensemble_size)
    losses: Array of shape (NUM_DEVICES, ensemble_size, num_epochs) -- the
      losses from each step of optimization.
    predictions: Array of num_samples samples of the BNF parameters from each of
    the (NUM_DEVICES, ensemble_size) surrogate posteriors.
  """
  features = jnp.array(features)
  target = jnp.array(target)

  if batch_size is not None:
    assert (
        target.shape[0] >= batch_size
    ), f'{batch_size=} exceeds {target.shape[0]=}'

  @functools.partial(jax.vmap, in_axes=(0, None, None))  # over MC samples
  @functools.partial(jax.vmap, in_axes=(0, None, None))  # over batch size
  def _target_log_prob_fn_inner(params, x_batch, y_batch):
    # NOTE: we divide by kl_weight here so that
    # tfp.vi.fit_surrogate_posterior_stateless minimizes:
    #     E_z~q [(log q(params) - _target_log_prob_fn(params)]
    #       = E_z~q [(log q(params) - log p(params))
    #                - neg_energy_fn(params, x, y) / kl_weight]
    # which is equivalent to minimizing:
    #     E_z~q [ kl_weight * (log q(params) - log p(params))
    #            - neg_energy_fn(params, x, y) ]
    return prior_d.log_prob(params) + (
        neg_energy_fn(params, x_batch, y_batch)
        * (target.shape[0] / y_batch.shape[0])
        / kl_weight
    )

  def _target_log_prob_fn(*params, seed):
    if batch_size is None:
      return _target_log_prob_fn_inner(params, features, target)
    batch = jax.random.permutation(
        seed, jnp.arange(target.shape[0]))[:batch_size]
    return _target_log_prob_fn_inner(params, features[batch], target[batch])

  def _make_surrogate_posterior(*params, batch_ndims=1):
    def _fn():
      for i in range(len(params) // 2):
        yield tfd.Normal(
            params[2 * i], 0.0001 + jax.nn.softplus(params[2 * i + 1])
        )

    return tfd.JointDistributionCoroutine(
        _fn, use_vectorized_map=True, batch_ndims=batch_ndims
    )

  init_seed, opt_seed = jax.random.split(seed, 2)
  num_devices = jax.device_count()
  init_params = make_vi_init(prior_d)(
      (num_devices, ensemble_size), seed=init_seed)

  @jax.pmap
  def _fit(init_params, fit_seed):
    surrogate_params, losses = tfp.vi.fit_surrogate_posterior_stateless(
        _target_log_prob_fn,
        _make_surrogate_posterior,
        init_params,
        optax.adam(learning_rate),
        num_epochs,
        sample_size=sample_size,
        jit_compile=False,
        seed=fit_seed,
    )
    return surrogate_params, losses

  @jax.pmap
  def _predict(surrogate_params, sample_seed):
    return _make_surrogate_posterior(*surrogate_params, batch_ndims=1).sample(
        num_samples, seed=sample_seed
    )

  fit_seed, sample_seed = jax.random.split(opt_seed, 2)
  surrogate_params, losses = _fit(
      init_params, jax.random.split(fit_seed, num_devices)
  )
  predictions = _predict(
      surrogate_params, jax.random.split(sample_seed, num_devices)
  )

  # Because we scaled the likelihood by `1 / kl_weight` in `_target_log_prob_fn
  # above (instead of scaling the KL(q||p) term) -- here we multiply the losses
  # by `kl_weight`.
  losses = np.transpose(losses, (0, 2, 1)) * kl_weight

  return (
      _make_surrogate_posterior(*surrogate_params, batch_ndims=2),
      losses,
      predictions,
  )
