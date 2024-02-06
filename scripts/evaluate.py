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

"""Evaluate ST-BNF."""

import json
import os
import time
from typing import Any, Union

from bayesnf import spatiotemporal
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp


tfd = tfp.distributions
ArrayT = jax.Array | np.ndarray


def drop_nan(x: ArrayT, y: ArrayT) -> tuple[ArrayT, ArrayT]:
  """Drop elements of x and y at indexes where y is NaN."""
  keep = ~np.isnan(y)
  return x[keep], y[keep]


def run_experiment(
    dataset: str,
    data_root: str,
    series_id: Union[int, str],
    output_dir: str,
    objective: str,
    dataset_config: dict[str, Any],
    model_config: dict[str, Any],
    inference_config: dict[str, Any],
    seed: jax.Array,
) -> tuple[ArrayT, ArrayT, ArrayT]:
  """Runs a single training experiment, writes output to output_dir."""
  path_train = os.path.join(data_root, f'{dataset}.{series_id}.train.csv')
  df_train = pd.read_csv(path_train, index_col=0, parse_dates=['datetime'])
  path_test = os.path.join(data_root, f'{dataset}.{series_id}.test.csv')
  df_test = pd.read_csv(path_test, index_col=0, parse_dates=['datetime'])

  os.makedirs(output_dir, exist_ok=True)
  path_model = os.path.join(
      output_dir, f'bnf-{objective}.{dataset}.{series_id}.json'
  )
  model_config.update(dict(
      feature_cols=dataset_config['feature_cols'],
      target_col=dataset_config['target_col'],
      timetype=dataset_config['timetype'],
      freq=dataset_config.get('freq', None),
      standardize=dataset_config.get('standardize', None),
  ))

  if objective == 'vi':
    base_cls = spatiotemporal.BayesianNeuralFieldVI
    objective_specific_inference_args = {
        'kl_weight': inference_config.get('kl_weight', 1.0),
        'sample_size': inference_config.get('sample_size', 10),
    }
  elif objective == 'map':
    base_cls = spatiotemporal.BayesianNeuralFieldMAP
    objective_specific_inference_args = {
        'num_splits': inference_config.get('num_particle_splits', 1),
    }
  elif objective == 'mle':
    base_cls = spatiotemporal.BayesianNeuralFieldMLE
    objective_specific_inference_args = {
        'num_splits': inference_config.get('num_particle_splits', 1),
    }
  else:
    raise ValueError(f'objective={objective}')

  start_time = time.perf_counter()
  inference_args = (
      dict(
          learning_rate=inference_config['learning_rate'],
          num_epochs=inference_config['num_epochs'],
          batch_size=inference_config.get('batch_size', None),
          ensemble_size=inference_config['num_particles'],
      )
      | objective_specific_inference_args
  )

  model = base_cls(**model_config).fit(df_train, seed, **inference_args)

  df_train_and_test = pd.concat([df_train, df_test])
  means, quantiles = model.predict(df_train_and_test,
                                   quantiles=(0.5, 0.025, 0.975))
  losses = model.losses_
  assert losses is not None
  runtime = time.perf_counter() - start_time

  path_log = path_model.replace('.json', '.log.json')
  with open(path_log, 'w') as f:
    log = {
        'dataset': dataset,
        'series_id': series_id,
        'runtime': runtime,
        'objective': objective,
        'dataset_config': dataset_config,
        'model_config': model_config,
        'inference_config': inference_config,
    }
    json.dump(log, f, indent=2, default=repr)

  path_loss = path_model.replace('.json', '.loss.csv')
  df_log = pd.DataFrame(losses.reshape((-1, losses.shape[-1])).T)
  with open(path_loss, 'w') as f:
    df_log.to_csv(f, index=False)

  pred_index = model.data_handler.copy_and_filter_table(df_train_and_test).index
  df_pred = pd.DataFrame(
      {
          'yhat': np.mean(means, axis=range(len(means.shape) - 1)),
          'yhat_p50': quantiles[0],
          'yhat_lower': quantiles[1],
          'yhat_upper': quantiles[2],
      },
      index=pred_index,
  )
  df_pred.sort_index(inplace=True)
  path_pred = path_model.replace('.json', '.pred.csv')
  with open(path_pred, 'w') as f:
    df_pred.to_csv(f, index=True)

  return losses, means, jnp.array(quantiles)
