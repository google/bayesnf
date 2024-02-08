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

"""Integration tests against golden data for evaluate.py."""

import io
import os
import pkgutil
import tempfile

import bayesnf
import dataset_config
import evaluate
import jax
import pandas as pd
import pytest

_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(name='golden_data_getter')
def fixture_golden_data_getter():
  def data_getter(fname):
    try:
      data = pd.read_csv(
          io.BytesIO(pkgutil.get_data('bayesnf.tests', f'test_data/{fname}')))
    except FileNotFoundError as exc:
      raise FileNotFoundError(
          f'Expected to see {fname} in {bayesnf.__file__}/tests/test_data'
      ) from exc
    return data
  return data_getter


def run_objective(objective, inference_config):
  dataset = 'chickenpox'
  series_id = '8'
  output_dir = tempfile.mkdtemp()
  fname = f'bnf-{objective}.chickenpox.8.pred.csv'
  _ = evaluate.run_experiment(
      dataset=dataset,
      data_root=os.path.join(_DIR, 'test_data'),
      series_id=series_id,
      output_dir=output_dir,
      objective=objective,
      dataset_config=dataset_config.DATASET_CONFIG[dataset],
      model_config=dataset_config.MODEL_CONFIG[dataset][objective],
      inference_config=inference_config,
      seed=jax.random.PRNGKey(0),
  )
  return pd.read_csv(os.path.join(output_dir, fname))


@pytest.mark.skip('Too slow.')
def test_map(golden_data_getter):
  inference_config = {
      'num_particles': 64,
      'num_epochs': 100,
      'learning_rate': 0.005}
  new_data = run_objective('map', inference_config)
  assert new_data.equals(golden_data_getter('bnf-map.chickenpox.8.pred.csv'))


@pytest.mark.skip('Too slow.')
def test_mle(golden_data_getter):
  inference_config = {
      'num_particles': 64,
      'num_epochs': 100,
      'learning_rate': 0.005}
  new_data = run_objective('mle', inference_config)
  assert new_data.equals(golden_data_getter('bnf-mle.chickenpox.8.pred.csv'))


@pytest.mark.skip('Too slow.')
def test_vi(golden_data_getter):
  inference_config = {
      'batch_size': 511,
      'kl_weight': 0.1,
      'learning_rate': 0.01,
      'num_epochs': 100,
      'num_particles': 64,
      'sample_size': 5}
  new_data = run_objective('vi', inference_config)
  assert new_data.equals(golden_data_getter('bnf-vi.chickenpox.8.pred.csv'))
