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

"""Run the baseline models on evaluation data."""

import os
import time
import types

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from absl import app
from absl import flags
from absl import logging
from scipy.cluster.vq import kmeans2
from tqdm import tqdm

from bayesnf.models import make_fourier_features
from bayesnf.models import make_seasonal_features

from dataset_config import DATASET_CONFIG
from dataset_config import MODEL_CONFIG


_DATA_ROOT = flags.DEFINE_string(
    'data_root', None, 'Location of input data.', required=True
)

_DATASET = flags.DEFINE_enum(
    'dataset',
    None,
    enum_values=DATASET_CONFIG.keys(),
    help='Dataset name',
    required=True)

_ALGORITHM = flags.DEFINE_enum(
    'algorithm',
    None,
    enum_values=['SVGP', 'ST-SVGP', 'MF-ST-SVGP', 'RF', 'GBOOST', 'TSREG'],
    help='Algorithm name',
    required=True)

_START_ID = flags.DEFINE_integer(
    'start_id', 5, 'Run experiments on series with IDs >= this value.')

_STOP_ID = flags.DEFINE_integer(
    'stop_id', None, 'Run experiments on series with IDs < this value.')

_SVGP_NUM_Z = flags.DEFINE_integer(
    'svgp_num_z', 2000, 'SVGP number of inducing points.')

_GBOOST_ESTIMATORS = flags.DEFINE_integer(
    'gboost_estimators', 100, 'Number of GBOOST estimators.')

_GBOOST_FEATURIZE = flags.DEFINE_boolean(
    'gboost_featurize', False, 'Add Fourier features to GBOOST baseline.')

_TSREG_METHOD = flags.DEFINE_enum(
    'tsreg_method',
    'OLS',
    enum_values=['OLS', 'RIDGE', 'LASSO'],
    help='Method for trend-surface regression.',
    required=False)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory.', required=True)

# pylint: disable=bad-whitespace
DATASET_CONFIG_BASELINE = {
    'air_quality': {
        'target_col'       : 'pm10',
        'timetype'         : 'unix',
        'feature_cols'     : ['datetime', 'latitude', 'longitude'],
        'standardize'      : ['datetime', 'latitude', 'longitude'],
        },
    'wind': {
        'target_col'       : 'wind',
        'timetype'         : 'unix',
        'feature_cols'     : ['datetime', 'latitude', 'longitude'],
        'standardize'      : ['datetime', 'latitude', 'longitude'],
        },
    'air': {
        'target_col'       : 'pm10',
        'timetype'         : 'unix',
        'feature_cols'     : ['datetime', 'latitude', 'longitude'],
        'standardize'      : ['datetime', 'latitude', 'longitude'],
        },
    'chickenpox': {
        'target_col'       : 'chickenpox',
        'timetype'         : 'unix',
        'feature_cols'     : ['datetime', 'latitude', 'longitude'],
        'standardize'      : ['datetime', 'latitude', 'longitude'],
        },
    'coprecip': {
        'target_col'       : 'ppt',
        'timetype'         : 'unix',
        'feature_cols'     : ['datetime', 'latitude', 'longitude'],
        'standardize'      : ['datetime', 'latitude', 'longitude'],
        },
    'sst': {
        'target_col'       : 'sst',
        'timetype'         : 'unix',
        'feature_cols'     : ['datetime', 'latitude', 'longitude', 'soi',],
        'standardize'      : ['datetime', 'latitude', 'longitude', 'soi',],
        },
}

ST_SVGP_CONFIG = {
    'air_quality': dict(
        len_space=0.2,
        ),
    'wind': dict(
        len_space=0.2,
        sparse=False,
        ),
    'air': dict(
        len_space=0.2,
        ),
    'chickenpox': dict(
        len_space=0.2,
        sparse=False,
        ),
    'coprecip': dict(
        len_space=0.2,
        sparse=True,
        iters=500,
        ),
}

SVGP_CONFIG = {
    'air_quality': dict(
        num_z_to_batch_size={
            1500: 400,
            2000: 600,
            2500: 800,
            5000: 2000,
            8000: 3000
            }),
    }

def get_dataset_tidy(
    root,
    dataset,
    series_id,
    *,
    feature_cols,
    target_col,
    timetype,
    freq=None,
    standardize=None,
    ):
  """Loads ((x_train, y_train), (x_test, y_test)) from tidy CSV file.

  Args:
    root: Name of root directory containing all datasets.
    dataset: Name of dataset to load.
    series_id: String or integer index for the train/test fold.
    feature_cols: Names of columns in dataset to use as regression features.
    target_col: Name of column in dataset to use as regression target.
    timetype: One of `['datetime', 'unix', 'index']`. Selecting `datetime`
      will keep the column named `datetime` intact, i.e., as a
      `datetime.datetime` object format. Selecting `unix` will
      convert to floating point UNIX epoch time.
      Selecting `index` will convert to integer index, with the first time
      stamp being zero.
    freq: If `timetype` is set to `'index'`, a Pandas frequency string
      compatible with the pandas `to_period` method. Default is `None`.
      Typical `freq` values include:
        - 'Y' for year
        - 'Q' for quarter
        - 'M' for month
        - 'D' for daily
        - 'H' for hour
        - 'T' for minute
        - 'S' for second
        - 'N' for nanosecond
        The `datetime` column will be discretized by invoking `to_period(freq)`,
        for additional details refer to
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_period.html
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.PeriodIndex.html
    standardize: List of feature columns to standardize to zero mean and unit
      variance. Default is `None`, for no features.
    root: CNS path of registry of datasets (optional).
  Returns:
    Tuple of numpy arrays ((x_train, y_train, (x_test, y_test))).
  """
  assert timetype in ['datetime', 'unix', 'index']
  assert freq is None or timetype in ['index']
  # Load training data.
  path_train = os.path.join(root, f'{dataset}.{series_id}.train.csv')
  with open(path_train, 'r') as f:
    df_train_raw = pd.read_csv(f, index_col=0, parse_dates=['datetime'])
  # Load testing data.
  path_test = os.path.join(root, f'{dataset}.{series_id}.test.csv')
  with open(path_test, 'r') as f:
    df_test_raw = pd.read_csv(f, index_col=0, parse_dates=['datetime'])
  # Copy datasets.
  df_train = df_train_raw.copy()
  df_test = df_test_raw.copy()
  # Convert datetime to numeric.
  if timetype == 'unix':
    df_train.datetime = df_train.datetime.astype('int64') // 1e9
    df_test.datetime = df_test.datetime.astype('int64') // 1e9
  if timetype == 'index':
    first_date = df_train.datetime.min().to_period(freq)
    df_train.datetime = df_train.datetime.dt.to_period(freq)
    df_test.datetime = df_test.datetime.dt.to_period(freq)
    df_train.datetime = (df_train.datetime - first_date).apply(lambda x: x.n)
    df_test.datetime = (df_test.datetime - first_date).apply(lambda x: x.n)
    logging.info(np.sort(df_test.datetime.unique()))
    logging.info(np.sort(df_train.datetime.unique()))
  # Extract training data.
  x_train = df_train[feature_cols].values
  y_train = df_train[target_col].values.astype(np.float64)
  # Extract testing data.
  x_test = df_test[feature_cols].values
  y_test = df_test[target_col].values.astype(np.float64)
  # Normalize features using training data.
  mu = np.zeros(len(feature_cols))
  std = np.ones(len(feature_cols))
  if standardize:
    idx = [feature_cols.index(f) for f in standardize]
    mu[idx] = np.mean(x_train[:, idx].astype(float), axis=0)
    std[idx] = np.std(x_train[:, idx].astype(float), axis=0)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std
  # Return train and test split.
  return types.SimpleNamespace(
      df_train=df_train_raw,
      df_test=df_test_raw,
      index_train=df_train.index,
      index_test=df_test.index,
      x_test=x_test,
      y_test=y_test,
      x_train=x_train,
      y_train=y_train,
      x_train_mu=mu,
      x_train_std=std,
      )

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  stop_id = _STOP_ID.value or DATASET_CONFIG[_DATASET.value]['num_series']
  for series_id in range(_START_ID.value, stop_id):
    logging.info('%s dataset %d', _DATASET.value, series_id)
    if _ALGORITHM.value == 'SVGP':
      run_experiment_gpflow(
          _DATA_ROOT.value,
          _DATASET.value,
          series_id,
          _SVGP_NUM_Z.value,
          **SVGP_CONFIG.get(_DATASET.value, {}),
          )
    elif _ALGORITHM.value in ['ST-SVGP', 'MF-ST-SVGP']:
      run_experiment_bayesnewton(
          _DATA_ROOT.value,
          _DATASET.value,
          series_id,
          method=_ALGORITHM.value,
          **ST_SVGP_CONFIG[_DATASET.value],
          )
    elif _ALGORITHM.value == 'RF':
      run_experiment_rf(
          _DATA_ROOT.value,
          _DATASET.value,
          series_id,
          )
    elif _ALGORITHM.value == 'GBOOST':
      run_experiment_gboost(
          _DATA_ROOT.value,
          _DATASET.value,
          series_id,
          n_estimators=_GBOOST_ESTIMATORS.value,
          featurize=_GBOOST_FEATURIZE.value,
          )
    elif _ALGORITHM.value == 'TSREG':
      run_experiment_tsreg(
          _DATA_ROOT.value,
          _DATASET.value,
          series_id,
          featurize=True,
          method=_TSREG_METHOD.value)
    else:
      raise ValueError(_ALGORITHM.value)


def run_experiment_bayesnewton(
    root,
    dataset,
    series_id,
    *,
    method,
    parallel=True,
    sparse=True,
    lr_newton=1.,
    lr_adam = 0.01,
    len_time=0.001,
    len_space=None,
    num_z_space=30,
    likelihood='gaussian',
    var_y=5.,
    iters=5000,  # Number of training steps (500 in the NeurIPS paper)
    binsize=None,
    ):
  """Runs a single bayesnewton baseline experiment."""
  # pylint:disable=invalid-name,g-import-not-at-top
  import bayesnewton
  import objax
  # pylint:enable=g-import-not-at-top

  assert method in ['ST-SVGP', 'MF-ST-SVGP']

  # Load data.
  table = get_dataset_tidy(
      root,
      dataset,
      series_id,
      feature_cols=DATASET_CONFIG_BASELINE[dataset]['feature_cols'],
      target_col=DATASET_CONFIG_BASELINE[dataset]['target_col'],
      timetype=DATASET_CONFIG_BASELINE[dataset]['timetype'],
      standardize=DATASET_CONFIG_BASELINE[dataset]['standardize'])

  # Standardize data.
  y_train_mu = np.nanmean(table.y_train)
  y_train_std = np.nanstd(table.y_train)
  y_train_norm = (table.y_train - y_train_mu) / y_train_std
  y_test_norm = (table.y_test - y_train_mu) / y_train_std

  # Data for training.
  (X, Y) = bnf.drop_nan(table.x_train, y_train_norm)
  t, R, Y = bnf.create_spatiotemporal_grid(X, Y)

  # Data for RMSE scoring.
  (X_test, Y_test) = bnf.drop_nan(table.x_test, table.y_test)
  t_test, R_test, Y_test = bnf.create_spatiotemporal_grid(X_test, Y_test)

  # Data for NLPD scoring.
  Y_test_norm = bnf.drop_nan(table.x_test, y_test_norm)[1]
  Y_test_norm = bnf.create_spatiotemporal_grid(X_test, Y_test_norm)[2]

  var_f = 1.
  opt_z = sparse
  z = kmeans2(R[0], num_z_space, minit='points')[0] if opt_z else R[0]

  logging.info('num time steps = %d', t.shape[0])
  logging.info('num spatial points = %d', R.shape[1])
  logging.info('num data points = %d', Y.size)

  kern_time = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_time)
  kern_space = bayesnewton.kernels.Separable([
      bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space),
      bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space),
      ])

  kern = bayesnewton.kernels.SpatioTemporalKernel(
      temporal_kernel=kern_time,
      spatial_kernel=kern_space,
      z=z,
      sparse=sparse,
      opt_z=opt_z,
      conditional='Full')

  likelihoods = {
      'gaussian': lambda: bayesnewton.likelihoods.Gaussian(variance=var_y),
      'poisson': lambda: bayesnewton.likelihoods.Poisson(binsize=binsize),
  }
  constructors = {
      'ST-SVGP': bayesnewton.models.MarkovVariationalGP,
      'MF-ST-SVGP': bayesnewton.models.MarkovVariationalMeanFieldGP,
  }
  model = constructors[method](
      kernel=kern,
      likelihood=likelihoods[likelihood](),
      X=t, R=R, Y=Y,
      parallel=parallel)

  opt_hypers = objax.optimizer.Adam(model.vars())
  energy = objax.GradValues(model.energy, model.vars())

  @objax.Function.with_vars(model.vars() + opt_hypers.vars())
  def train_op():
    # perform inference and update variational params
    model.inference(lr=lr_newton)
    # compute energy and its gradients w.r.t. hypers
    dE, (E,) = energy()
    opt_hypers(lr_adam, dE)
    return E

  train_op = objax.Jit(train_op)

  def compute_metrics(model):
    posterior_mean, posterior_var = model.predict_y(X=t_test, R=R_test)
    posterior_mean = posterior_mean * y_train_std + y_train_mu
    nlpd = model.negative_log_predictive_density(
        X=t_test, R=R_test, Y=Y_test_norm
    )
    rmse = np.sqrt(np.nanmean(np.square(posterior_mean - np.squeeze(Y_test))))
    return (rmse, nlpd)

  n_ckpt = int(1 + np.ceil(np.log2(iters)))
  epoch = np.zeros(n_ckpt, dtype=int)
  elapsed = 0
  nlpd = np.zeros(n_ckpt)
  rmse = np.zeros(n_ckpt)
  elbos = np.zeros(n_ckpt)
  runtime = np.zeros(n_ckpt)
  (c, ckpt) = (0, 1)
  progbar = tqdm(range(iters))
  for i in progbar:
    start = time.time()
    loss = train_op()
    elapsed += time.time() - start
    if (i + 1 == ckpt) or (i == iters - 1):
      epoch[c] = i
      runtime[c] = elapsed
      elbos[c] = loss
      rmse[c], nlpd[c] = compute_metrics(model)
      logging.info(
          'epoch=%s, loss=%.2f, rmse=%.2f, nlpd=%.2f, runtime=%.2f',
          epoch[c], elbos[c], rmse[c], nlpd[c], runtime[c])
      progbar.set_postfix(loss=elbos[c], rmse=rmse[c], nlpd=nlpd[c])
      (c, ckpt) = (c+1, 2*ckpt)

  # Write log history.
  df = pd.DataFrame(dict(epoch=epoch, runtime=runtime, rmse=rmse, nlpd=nlpd))
  Path(_OUTPUT_DIR.value).mkdir(parents=True, exist_ok=True)
  prefix = method.lower()
  csv_path_log = os.path.join(
      _OUTPUT_DIR.value,
      f'bayesnewton-{prefix}.{dataset}.{series_id}.log.csv')
  with open(csv_path_log, 'w') as f:
    df.to_csv(f, index=False)
  logging.info(csv_path_log)

  # Write forecasts.
  # TODO(ffsaad): Implement correctly for non-Gaussian likelihood.
  # -- Prepare the probe data frame.
  index_probe = np.concatenate((table.index_train, table.index_test))
  x_probe = np.concatenate((table.x_train, table.x_test))
  y_probe = np.concatenate((table.y_train, table.y_test))
  t_probe, R_probe, _ = bnf.create_spatiotemporal_grid(x_probe, y_probe)
  df_probe = pd.DataFrame(x_probe, index=index_probe)
  df_probe.index.name = '__index__'
  df_probe.reset_index(inplace=True)
  # -- Obtain predictions.
  mean_f, var_f = model.predict_y(t_probe, R=R_probe)
  yhat_mean = np.ravel(mean_f) * y_train_std + y_train_mu
  yhat_std = np.sqrt(np.ravel(var_f)) * y_train_std
  dist = scipy.stats.norm(loc=yhat_mean, scale=yhat_std)
  yhat_lower = dist.ppf(.025)
  yhat_upper = dist.ppf(.975)
  df_pred = pd.DataFrame(
      np.column_stack((
          np.repeat(t_probe, R_probe.shape[1]),
          np.row_stack(R_probe),
          yhat_mean,
          yhat_std,
          yhat_lower,
          yhat_upper,
      )))
  # -- Merge predictions into the probe data frame.
  # -- TODO(ffsaad): Add distance of each location to nearest inducing point.
  merge_on = list(df_probe.columns[1:])
  df_join = pd.merge(df_probe, df_pred, on=merge_on, how='left', validate='1:1')
  df_join.set_index('__index__', inplace=True)
  df_join.index.name = None
  df_join.drop(columns=merge_on, inplace=True)
  df_join.columns = ['yhat', 'yhat_std', 'yhat_lower', 'yhat_upper']
  # -- Write to disk.
  csv_path_predictions = csv_path_log.replace('.log.', '.pred.')
  with open(csv_path_predictions, 'w') as f:
    df_join.to_csv(f, index=True)
  logging.info(csv_path_predictions)
  # pylint:enable=invalid-name


# From the ST-SVGP paper (Table 3):
# To ensure the run times are comparable on both CPU and GPU,
# we run SVGP with inducing points (mini batch size):
# 2000(600), 2500(800), 5000(2000), and 8000(3000)
def run_experiment_gpflow(
    root,
    dataset,
    series_id,
    num_z,
    *,
    num_z_to_batch_size,
    kernel_lengthscales=(.01, .2, .2),
    natgrad_step_size=1.0,
    likelihood='gaussian',
    likelihood_noise=5.0,  # gaussian
    binsize=1.,  # poisson
    ):
  """Runs a single instance of SVGP optimization."""
  # pylint:disable=invalid-name,g-import-not-at-top
  import GPflow
  from GPflow.optimizers import NaturalGradient
  from GPflow.utilities import set_trainable
  import tensorflow as tf
  # pylint:enable=g-import-not-at-top
  batch_size = num_z_to_batch_size[num_z]

  table = get_dataset_tidy(
      root,
      dataset,
      series_id,
      feature_cols=DATASET_CONFIG_BASELINE[dataset]['feature_cols'],
      target_col=DATASET_CONFIG_BASELINE[dataset]['target_col'],
      timetype=DATASET_CONFIG_BASELINE[dataset]['timetype'],
      standardize=DATASET_CONFIG_BASELINE[dataset]['standardize'])
  (x_train, y_train) = bnf.drop_nan(table.x_train, table.y_train)
  (x_test, y_test) = bnf.drop_nan(table.x_test, table.y_test)

  logging.info('x: %s', x_train.shape)

  kernel_variances = 1.0
  epochs = 300  # 300, 500, etc (minibatches) in original code
  step_size = 0.01

  N, D = x_train.shape
  logging.info(
      f'N: {N}, batch_size: {batch_size}, '
      f'num_z: {num_z}, N_test: {x_test.shape[0]}')
  z_all = kmeans2(x_train, num_z, minit='points')[0]

  kernel = GPflow.kernels.Matern32

  k = None
  for d in range(D):
    if isinstance(kernel_lengthscales, list):
      k_ls = kernel_lengthscales[d]
    else:
      k_ls = kernel_lengthscales
    if isinstance(kernel_variances, list):
      k_var = kernel_variances[d]
    else:
      k_var = kernel_variances
    k_d = kernel(
        lengthscales=[k_ls],
        variance=k_var,
        active_dims=[d])
    if k is None:
      k = k_d
    else:
      k = k * k_d

  init_as_cvi = True
  if init_as_cvi:
    M = z_all.shape[0]
    jit = 1e-6
    Kzz = k(z_all, z_all)
    def inv(K):
      K_chol = scipy.linalg.cholesky(K + jit * np.eye(M), lower=True)
      return scipy.linalg.cho_solve((K_chol, True), np.eye(K.shape[0]))
    # manual q(u) decomposition
    nat1 = np.zeros([M, 1])
    nat2 = -0.5 * inv(Kzz)
    lam1 = 1e-5 * np.ones([M, 1])
    lam2 = -0.5 * np.eye(M)
    S = inv(-2 * (nat2 + lam2))
    m = S @ (lam1 + nat1)
    S_chol = scipy.linalg.cholesky(S + jit * np.eye(M), lower=True)
    # S_flattened = S_chol[np.tril_indices(M, 0)]
    q_mu = m
    q_sqrt = np.array([S_chol])
  else:
    q_mu = 1e-5 * np.ones([z_all.shape[0], 1])  # match gpjax init
    q_sqrt = None

  likelihoods = {
      'gaussian': GPflow.likelihoods.Gaussian(variance=likelihood_noise),
      'poisson': GPflow.likelihoods.Poisson(binsize=binsize),
  }
  lik = likelihoods[likelihood]
  data = (x_train, y_train)

  m = GPflow.models.SVGP(
      inducing_variable=z_all,
      whiten=True,
      kernel=k,
      mean_function=None,
      likelihood=lik,
      q_mu=q_mu,
      q_sqrt=q_sqrt)

  set_trainable(m.inducing_variable, True)

  train_dataset = (
      tf.data.Dataset
      .from_tensor_slices(data)
      .repeat()
      .shuffle(N)
      .batch(batch_size)
      )
  train_iter = iter(train_dataset)
  training_loss = m.training_loss_closure(train_iter, compile=True)

  # Make it so adam does not train these
  set_trainable(m.q_mu, False)
  set_trainable(m.q_sqrt, False)
  natgrad_opt = NaturalGradient(gamma=natgrad_step_size)
  variational_params = [(m.q_mu, m.q_sqrt)]
  optimizer = tf.optimizers.Adam
  adam_opt_for_vgp = optimizer(step_size)
  loss_arr = []

  def _prediction_fn(x_, y_):
    # Poisson likelihood uses quadrature, which OOMs on GPU.
    with tf.device('CPU:0' if likelihood == 'poisson' else 'GPU:0'):
      mu, var = m.predict_y(x_)
      log_pred_density = m.predict_log_density((x_, y_))
    return mu.numpy(), var.numpy(), log_pred_density.numpy()

  iter_counts = []
  nlpd = []
  rmse = []
  runtimes = []

  elbo_fn = tf.function(m.elbo, jit_compile=True)

  @tf.function
  def train_step():
    # NAT GRAD STEP
    natgrad_opt.minimize(training_loss, var_list=variational_params)
    # ADAM STEP
    adam_opt_for_vgp.minimize(training_loss, var_list=m.trainable_variables)
    return tf.zeros([])

  # MINIBATCHING TRAINING
  t0 = time.time()
  niters = epochs * N // batch_size
  ckpt = max(1, int(N // (batch_size * 10)))  # metrics ~10x/epoch
  progbar = tqdm(range(niters))
  metrics_time = 0.
  for i in progbar:
    train_step()
    if ((i % ckpt) == 0) or (i == niters - 1):
      # Compute per-step metrics.
      mt0 = time.time()
      iter_counts.append(i + 1)
      runtimes.append(time.time() - t0 - metrics_time)
      posterior_mean, _, lpd = _prediction_fn(x_test, y_test)
      nlpd.append(-np.mean(lpd))
      rmse.append(np.sqrt(
          np.nanmean((np.squeeze(y_test) - np.squeeze(posterior_mean))**2)))
      loss_arr.append(-elbo_fn(data).numpy())
      progbar.set_postfix(loss=loss_arr[-1], nlpd=nlpd[-1], rmse=rmse[-1])
      metrics_time += time.time() - mt0

  t1 = time.time()
  avg_iter_time = (t1 - t0 - metrics_time) / niters
  logging.info('average iter time: %2.2f secs', avg_iter_time)
  avg_epoch_time = (t1 - t0 - metrics_time) / epochs
  logging.info('average epoch time: %2.2f secs', avg_epoch_time)
  # pylint:enable=invalid-name

  # Write log history.
  df = pd.DataFrame(dict(
      epoch=np.array(iter_counts) * batch_size / N,
      runtime=np.array(runtimes),
      rmse=np.array(rmse),
      nlpd=np.array(nlpd)))
  Path(_OUTPUT_DIR.value).mkdir(parents=True, exist_ok=True)
  method = f'gpflow-svgp-{num_z}-{batch_size}'
  csv_path_log = os.path.join(
      _OUTPUT_DIR.value,
      f'{method}.{dataset}.{series_id}.csv')
  with open(csv_path_log, 'w') as f:
    df.to_csv(f, index=False)
  logging.info('Wrote results to %s', csv_path_log)

  # Write forecasts.
  # TODO(ffsaad): Implement correctly for non-Gaussian likelihood.
  # -- Prepare the probe data frame.
  index_probe = np.concatenate((table.index_train, table.index_test))
  x_probe = np.concatenate((table.x_train, table.x_test))
  y_probe = np.concatenate((table.y_train, table.y_test))
  yhat_mean, yhat_var, yhat_nlpd = _prediction_fn(x_probe, y_probe)
  yhat_std = np.sqrt(yhat_var)
  dist = scipy.stats.norm(loc=yhat_mean, scale=yhat_std)
  yhat_lower = dist.ppf(.025)
  yhat_upper = dist.ppf(.975)
  df_pred = pd.DataFrame({
      'yhat': yhat_mean,
      'yhat_std': yhat_std,
      'yhat_lower': yhat_lower,
      'yhat_upper': yhat_upper,
      }, index=index_probe)
  df_pred.sort_index(inplace=True)
  csv_path_pred = csv_path_log.replace('.log.', '.pred.')
  with open(csv_path_pred, 'w') as f:
    df_pred.to_csv(f, index=True)
  logging.info(csv_path_pred)


def run_experiment_rf(
    root,
    dataset,
    series_id,
    ):
  """Runs a single RandomForest baseline experiment."""
  # pylint:disable=invalid-name,g-import-not-at-top
  from sklearn.ensemble import RandomForestRegressor
  # pylint:enable=g-import-not-at-top
  table = get_dataset_tidy(
      root,
      dataset,
      series_id,
      feature_cols=DATASET_CONFIG_BASELINE[dataset]['feature_cols'],
      target_col=DATASET_CONFIG_BASELINE[dataset]['target_col'],
      timetype=DATASET_CONFIG_BASELINE[dataset]['timetype'],
      standardize=DATASET_CONFIG_BASELINE[dataset]['standardize'],
      )
  (x_train, y_train) = bnf.drop_nan(table.x_train, table.y_train)
  (x_test, y_test) = bnf.drop_nan(table.x_test, table.y_test)
  start = time.time()
  regressor = RandomForestRegressor().fit(x_train, y_train)
  runtime = time.time() - start

  # Write log history.
  yhat = regressor.predict(x_test)
  rmse = np.sqrt(np.nanmean((np.squeeze(y_test) - np.squeeze(yhat)) ** 2))
  df = pd.DataFrame(
      dict(epoch=[0], runtime=[runtime], rmse=[rmse], nlpd=[np.nan])
  )
  Path(_OUTPUT_DIR.value).mkdir(parents=True, exist_ok=True)
  csv_path_log = os.path.join(
      _OUTPUT_DIR.value, f'rf.{dataset}.{series_id}.log.csv'
  )
  with open(csv_path_log, 'w') as f:
    df.to_csv(f, index=False)
  logging.info('Wrote results to %s', csv_path_log)

  # Write forecasts.
  index_probe = np.concatenate((table.index_train, table.index_test))
  x_probe = np.concatenate((table.x_train, table.x_test))
  y_probe = np.concatenate((table.y_train, table.y_test))
  yhat_mean = regressor.predict(x_probe)
  df_pred = pd.DataFrame({
      'yhat': yhat_mean,
      'yhat_std': np.repeat(0, yhat_mean.shape[0]),
      'yhat_lower': yhat_mean,
      'yhat_upper': yhat_mean,
      }, index=index_probe)
  df_pred.sort_index(inplace=True)
  csv_path_pred = csv_path_log.replace('.log.', '.pred.')
  with open(csv_path_pred, 'w') as f:
    df_pred.to_csv(f, index=True)
  logging.info(csv_path_pred)


def featurize_inputs(
    x, seasonality_periods, num_seasonal_harmonics, fourier_degrees):
  seasonal_features = make_seasonal_features(
      x[:, 0], seasonality_periods, num_seasonal_harmonics, rescale=False
  )
  fourier_features = [
      make_fourier_features(x[:, i], degree, True)
      for i, degree in enumerate(fourier_degrees)
  ]
  features = [
      *[u.T[:, np.newaxis] for u in x.T],
      *fourier_features,
      seasonal_features,
  ]
  return np.column_stack(features)


def run_experiment_gboost(
    root,
    dataset,
    series_id,
    *,
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=9,
    min_samples_split=9,
    learning_rate=0.05,
    featurize=True,
    ):
  """Runs a single RandomForest baseline experiment."""
  # pylint:disable=invalid-name,g-import-not-at-top
  from sklearn.ensemble import GradientBoostingRegressor
  # pylint:enable=g-import-not-at-top
  if not featurize:
    table = get_dataset_tidy(
        root,
        dataset,
        series_id,
        feature_cols=DATASET_CONFIG_BASELINE[dataset]['feature_cols'],
        target_col=DATASET_CONFIG_BASELINE[dataset]['target_col'],
        timetype=DATASET_CONFIG_BASELINE[dataset]['timetype'],
        standardize=DATASET_CONFIG_BASELINE[dataset]['standardize'],
        )
    x_train = table.x_train
    x_test = table.x_test
  else:
    table = get_dataset_tidy(
        root,
        dataset,
        series_id,
        feature_cols=DATASET_CONFIG[dataset]['feature_cols'],
        target_col=DATASET_CONFIG[dataset]['target_col'],
        timetype=DATASET_CONFIG[dataset]['timetype'],
        freq=DATASET_CONFIG[dataset]['freq'],
        standardize=DATASET_CONFIG[dataset]['standardize'],
    )
    (x_train, x_test) = [
        featurize_inputs(
            z,
            MODEL_CONFIG[dataset]['map']['seasonality_periods'],
            MODEL_CONFIG[dataset]['map']['num_seasonal_harmonics'],
            4 * np.ones(table.x_train.shape[1]),
        )
        for z in (table.x_train, table.x_test)
    ]

  (x_train_drop, y_train_drop) = bnf.drop_nan(x_train, table.y_train)
  (x_test_drop, y_test_drop) = bnf.drop_nan(x_test, table.y_test)

  models = {}
  common_params = dict(
      learning_rate=learning_rate,
      n_estimators=n_estimators,
      max_depth=max_depth,
      min_samples_leaf=min_samples_leaf,
      min_samples_split=min_samples_split,
  )
  start = time.time()
  for alpha in [0.025, 0.5, 0.975]:
    logging.info('Fitting alpha=%f', alpha)
    gbr = GradientBoostingRegressor(
        loss='quantile', alpha=alpha, **common_params
    )
    models[alpha] = gbr.fit(x_train_drop, y_train_drop)
    logging.info('Elapsed: %f', time.time() - start)
  runtime = time.time() - start

  # Write log history.
  yhat = models[0.5].predict(x_test_drop)
  rmse = np.sqrt(np.nanmean((np.squeeze(y_test_drop) - np.squeeze(yhat)) ** 2))
  df = pd.DataFrame(
      dict(epoch=[0], runtime=[runtime], rmse=[rmse], nlpd=[np.nan])
  )
  Path(_OUTPUT_DIR.value).mkdir(parents=True, exist_ok=True)
  csv_path_log = os.path.join(
      _OUTPUT_DIR.value,
      f'gboost-{n_estimators}-{featurize}.{dataset}.{series_id}.log.csv',
  )
  with open(csv_path_log, 'w') as f:
    df.to_csv(f, index=False)
  logging.info('Wrote results to %s', csv_path_log)

  # Write forecasts.
  index_probe = np.concatenate((table.index_train, table.index_test))
  x_probe = np.concatenate((x_train, x_test))
  y_probe = np.concatenate((table.y_train, table.y_test))
  yhat_mean = models[0.5].predict(x_probe)
  yhat_lower = models[0.025].predict(x_probe)
  yhat_upper = models[0.975].predict(x_probe)
  df_pred = pd.DataFrame({
      'yhat': yhat_mean,
      'yhat_std': np.repeat(0, yhat_mean.shape[0]),
      'yhat_lower': yhat_lower,
      'yhat_upper': yhat_upper,
      }, index=index_probe)
  df_pred.sort_index(inplace=True)
  csv_path_pred = csv_path_log.replace('.log.', '.pred.')
  with open(csv_path_pred, 'w') as f:
    df_pred.to_csv(f, index=True)
  logging.info(csv_path_pred)


def run_experiment_tsreg(
    root,
    dataset,
    series_id,
    *,
    featurize=True,
    method=None,
    ):
  """Runs Trend-Surface Regression baseline experiment."""
  # pylint:disable=invalid-name,g-import-not-at-top
  import sklearn.linear_model
  # pylint:enable=g-import-not-at-top
  if not featurize:
    table = get_dataset_tidy(
        root,
        dataset,
        series_id,
        feature_cols=DATASET_CONFIG_BASELINE[dataset]['feature_cols'],
        target_col=DATASET_CONFIG_BASELINE[dataset]['target_col'],
        timetype=DATASET_CONFIG_BASELINE[dataset]['timetype'],
        standardize=DATASET_CONFIG_BASELINE[dataset]['standardize'],
        )
    x_train = table.x_train
    x_test = table.x_test
  else:
    table = get_dataset_tidy(
        root,
        dataset,
        series_id,
        feature_cols=DATASET_CONFIG[dataset]['feature_cols'],
        target_col=DATASET_CONFIG[dataset]['target_col'],
        timetype=DATASET_CONFIG[dataset]['timetype'],
        freq=DATASET_CONFIG[dataset]['freq'],
        standardize=DATASET_CONFIG[dataset]['standardize'],
    )
    (x_train, x_test) = [
        featurize_inputs(
            z,
            MODEL_CONFIG[dataset]['map']['seasonality_periods'],
            MODEL_CONFIG[dataset]['map']['num_seasonal_harmonics'],
            4 * np.ones(table.x_train.shape[1]),
        )
        for z in (table.x_train, table.x_test)
    ]

  (x_train_drop, y_train_drop) = bnf.drop_nan(x_train, table.y_train)
  (x_test_drop, y_test_drop) = bnf.drop_nan(x_test, table.y_test)

  # Fit regression.
  if method == 'OLS':
    model = sklearn.linear_model.LinearRegression()
  elif method == 'RIDGE':
    model = sklearn.linear_model.Ridge()
  elif method == 'LASSO':
    model = sklearn.linear_model.Lasso()
  else:
    raise ValueError(f'Unknown TSREG method: {method}')
  start = time.time()
  model.fit(x_train_drop, y_train_drop)
  runtime = time.time() - start

  # Compute RSS.
  yhat_train = model.predict(x_train_drop)
  rss = np.sum(np.square(yhat_train - y_train_drop))
  yhat_std = np.sqrt(rss / (x_train_drop.shape[0] - x_train_drop.shape[1]))

  # Write log history.
  yhat = model.predict(x_test_drop)
  rmse = np.sqrt(np.nanmean((np.squeeze(y_test_drop) - np.squeeze(yhat)) ** 2))
  df = pd.DataFrame(
      dict(epoch=[0], runtime=[runtime], rmse=[rmse], nlpd=[np.nan])
  )
  Path(_OUTPUT_DIR.value).mkdir(parents=True, exist_ok=True)
  csv_path_log = os.path.join(
      _OUTPUT_DIR.value, f'tsreg-{method}.{dataset}.{series_id}.log.csv'
  )
  with open(csv_path_log, 'w') as f:
    df.to_csv(f, index=False)
  logging.info('Wrote results to %s', csv_path_log)

  # Write forecasts.
  index_probe = np.concatenate((table.index_train, table.index_test))
  x_probe = np.concatenate((x_train, x_test))
  y_probe = np.concatenate((table.y_train, table.y_test))
  yhat_mean = model.predict(x_probe)
  dist = scipy.stats.norm(loc=yhat_mean, scale=yhat_std)
  yhat_lower = dist.ppf(.025)
  yhat_upper = dist.ppf(.975)
  df_pred = pd.DataFrame({
      'yhat': yhat_mean,
      'yhat_std': np.repeat(yhat_std, yhat_mean.shape[0]),
      'yhat_lower': yhat_lower,
      'yhat_upper': yhat_upper,
      }, index=index_probe)
  df_pred.sort_index(inplace=True)
  csv_path_pred = csv_path_log.replace('.log.', '.pred.')
  with open(csv_path_pred, 'w') as f:
    df_pred.to_csv(f, index=True)
  logging.info(csv_path_pred)


if __name__ == '__main__':
  app.run(main)
