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

r"""Run BNF time-series analysis experiments on M3 dataset."""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from bayesnf import dataset_config
from bayesnf import evaluate
import jax


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory.', required=True
)

_DATA_ROOT = flags.DEFINE_string(
    'data_root', None, 'Location of input data.', required=True
)

_DATASET = flags.DEFINE_enum(
    'dataset',
    None,
    enum_values=dataset_config.DATASET_CONFIG.keys(),
    help='Dataset name',
    required=True,
)

_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    default='map',
    enum_values=['map', 'mle', 'vi'],
    help='Training objective',
)

_START_ID = flags.DEFINE_integer(
    'start_id', 0, 'Run experiments on series with IDs >= this value.'
)

_STOP_ID = flags.DEFINE_integer(
    'stop_id', None, 'Run experiments on series with IDs < this value.'
)

_NUM_PARTICLES = flags.DEFINE_integer(
    'num_particles', None, 'Override the number of particles for inference.'
)


# pylint: disable=bad-whitespace
def _get_inference_config():
  """Return inference configs."""
  ret = {}

  ret['air_quality'] = {
      'map': {
          'num_particles': 16,
          'num_epochs': 4000,
          'learning_rate': 0.005,
          'batch_size': 38096,
      },
      'vi': {
          'num_particles': 16,
          'num_epochs': 500,
          'learning_rate': 0.01,
          'batch_size': 3500,
          'kl_weight': 0.2,
          'sample_size': 5,
      },
  }
  ret['air_quality']['mle'] = ret['air_quality']['map']

  ret['wind'] = {
      'map': {
          'num_particles': 64,
          'num_epochs': 10000,
          'learning_rate': 0.005,
      },
      'vi': {
          'num_particles': 64,
          'num_epochs': 2000,
          'learning_rate': 0.01,
          'batch_size': 3944,
          'kl_weight': 0.1,
          'sample_size': 5,
      },
  }
  ret['wind']['mle'] = ret['wind']['map']

  ret['air'] = {
      'map': {
          'num_particles': 8,
          'num_epochs': 7500,
          'learning_rate': 0.005,
      },
      'vi': {
          'num_particles': 8,
          'num_epochs': 1000,
          'learning_rate': 0.01,
          'batch_size': 3800,
          'kl_weight': 0.2,
          'sample_size': 5,
      },
  }
  ret['air']['mle'] = ret['air']['map']

  ret['chickenpox'] = {
      'map': {
          'num_particles': 64,
          'num_epochs': 10000,
          'learning_rate': 0.005,
      },
      'vi': {
          'num_particles': 64,
          'num_epochs': 1000,
          'learning_rate': 0.01,
          'batch_size': 511,
          'kl_weight': 0.1,
          'sample_size': 5,
      },
  }
  ret['chickenpox']['mle'] = ret['chickenpox']['map']

  ret['coprecip'] = {
      'map': {
          'num_particles': 16,
          'num_epochs': 7500,
          'learning_rate': 0.005,
      },
      'vi': {
          'num_particles': 16,
          'num_epochs': 750,
          'learning_rate': 0.01,
          'batch_size': 3300,
          'kl_weight': 0.2,
          'sample_size': 5,
      },
  }
  ret['coprecip']['mle'] = ret['coprecip']['map']

  ret['sst'] = {
      'map': {
          'num_particles': 16,
          'num_epochs': 5000,
          'learning_rate': 0.005,
          'batch_size': 221127,
      },
      'vi': {
          'num_particles': 16,
          'num_epochs': 600,
          'learning_rate': 0.005,
          'batch_size': 8845,
          'kl_weight': 0.5,
          'sample_size': 5,
      },
  }
  ret['sst']['mle'] = ret['sst']['map']

  ret['M3Month'] = {
      'map': {
          'num_particles': 64,
          'num_epochs': 5000,
          'learning_rate': 0.01,
      },
  }
  ret['M3Month']['mle'] = ret['M3Month']['map']

  return ret


# pylint: enable=bad-whitespace

INFERENCE_CONFIG = _get_inference_config()


def main(argv: Sequence[str]):
  if len(argv) > 3:
    raise app.UsageError('Too many command-line arguments.')

  if _NUM_PARTICLES.value:
    for k in INFERENCE_CONFIG:
      for obj in INFERENCE_CONFIG[k]:
        INFERENCE_CONFIG[k][obj]['num_particles'] = _NUM_PARTICLES.value

  stop_id = (
      _STOP_ID.value
      or dataset_config.DATASET_CONFIG[_DATASET.value]['num_series']
  )
  for series_id in range(_START_ID.value, stop_id):
    logging.info('%s series_id %d', _DATASET.value, series_id)
    evaluate.run_experiment(
        dataset=_DATASET.value,
        data_root=_DATA_ROOT.value,
        series_id=dataset_config.DATASET_CONFIG[_DATASET.value][
            'series_id_fmt'](series_id),
        output_dir=_OUTPUT_DIR.value,
        objective=_OBJECTIVE.value,
        dataset_config=dataset_config.DATASET_CONFIG[_DATASET.value],
        model_config=dataset_config.MODEL_CONFIG[_DATASET.value][
            _OBJECTIVE.value
        ],
        inference_config=INFERENCE_CONFIG[_DATASET.value][_OBJECTIVE.value],
        seed=jax.random.PRNGKey(2023100400 + int(series_id)),
    )


if __name__ == '__main__':
  app.run(main)
