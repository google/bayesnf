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

r"""Launch spatiotemporal BNF experiments."""

from absl import app
from absl import flags

from xmanager import xm
from xmanager import xm_local


DATASET_SIZE = {
    'air_quality': 12,
    'wind': 12,
    'air': 12,
    'chickenpox': 12,
    'coprecip': 12,
    'sst': 12,
    'M3Month': 1428,
}

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory.', required=True)

_START_ID = flags.DEFINE_integer(
    'start_id', 0, 'Run experiments on series with IDs >= this value.')

_STOP_ID = flags.DEFINE_integer(
    'stop_id', -1, 'Run experiments on series with IDs < this value.')

_DATASET = flags.DEFINE_enum(
    'dataset',
    default=None,
    enum_values=list(DATASET_SIZE),
    help='Dataset name',
    required=True,
)

_OBJECTIVE = flags.DEFINE_enum(
    'objective',
    default='map',
    enum_values=['map', 'mle', 'vi'],
    help='Training objective',
)

_NUM_WORK_UNITS = flags.DEFINE_integer(
    'num_work_units', 1, 'Divide the experiments into this many work units.'
)

_NUM_PARTICLES = flags.DEFINE_integer(
    'num_particles', None, 'Override the number of particles for inference.')


def main(_):
  with xm_local.create_experiment(
      experiment_title=f'Spatiotemporal BNF-{_OBJECTIVE.value} {_DATASET.value}'
  ) as experiment:
    # Packaging.
    [executable] = experiment.package([
        xm.bazel_binary(
            label='//research/probability/bnf/opensource:runner',
            executor_spec=xm_local.Vertex.Spec(),
        ),
    ])

    # Execution.
    requirements = xm.JobRequirements(tpu_v5='2x2')

    # Start and stop ID.
    start_id = _START_ID.value
    stop_id = _STOP_ID.value
    if (stop_id == - 1) or (stop_id > DATASET_SIZE[_DATASET.value]):
      stop_id = DATASET_SIZE[_DATASET.value]
    assert start_id < stop_id, f'No experiments: ({start_id=}, {stop_id=}).'

    # Split among work units.
    splits = [
        (i * (stop_id - start_id)) // _NUM_WORK_UNITS.value + start_id
        for i in range(_NUM_WORK_UNITS.value + 1)
    ]
    for i in range(_NUM_WORK_UNITS.value):
      experiment.add(
          xm.Job(
              executable,
              args={
                  'output_dir': _OUTPUT_DIR.value,
                  'dataset': _DATASET.value,
                  'start_id': splits[i],
                  'stop_id': splits[i+1],
                  'objective': _OBJECTIVE.value,
                  'num_particles': _NUM_PARTICLES.value,
              },
              executor=xm_local.Vertex(requirements=requirements)))


if __name__ == '__main__':
  app.run(main)
