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


DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(name='golden_data_getter')
def fixture_golden_data_getter():
  def data_getter(fname):
    data_path = os.path.join(DIR, 'test_data', fname)
    return pd.read_csv(data_path, index_col=0)
  return data_getter


def run_objective(objective, inference_config):
  dataset = 'chickenpox'
  series_id = '8'
  output_dir = tempfile.mkdtemp()
  fname = f'bnf-{objective}.chickenpox.8.pred.csv'
  evaluate.run_experiment(
      dataset=dataset,
      data_root=os.path.join(DIR, 'test_data'),
      series_id=series_id,
      output_dir=output_dir,
      objective=objective,
      dataset_config=dataset_config.DATASET_CONFIG[dataset],
      model_config=dataset_config.MODEL_CONFIG[dataset][objective],
      inference_config=inference_config,
      seed=jax.random.PRNGKey(0),
  )
  return pd.read_csv(os.path.join(output_dir, fname), index_col=0)


@pytest.mark.skip(reason='Github server version issue.')
def test_map_mini(golden_data_getter):
  inference_config = {
      'num_particles': 4,
      'num_epochs': 5,
      'learning_rate': 0.005}
  new_data = run_objective('map', inference_config)
  old_data = golden_data_getter('bnf-map.chickenpox.8.mini.pred.csv')
  assert new_data.equals(old_data)


@pytest.mark.skip(reason='Github server version issue.')
def test_mle_mini(golden_data_getter):
  inference_config = {
      'num_particles': 4,
      'num_epochs': 5,
      'learning_rate': 0.005}
  new_data = run_objective('mle', inference_config)
  old_data = golden_data_getter('bnf-mle.chickenpox.8.mini.pred.csv')
  assert new_data.equals(old_data)


@pytest.mark.skip(reason='Github server version issue.')
def test_vi_mini(golden_data_getter):
  inference_config = {
      'batch_size': None,
      'kl_weight': 0.1,
      'learning_rate': 0.01,
      'num_epochs': 2,
      'num_particles': 1,
      'sample_size_divergence': 5}
  new_data = run_objective('vi', inference_config)
  old_data = golden_data_getter('bnf-vi.chickenpox.8.mini.pred.csv')
  assert new_data.equals(old_data)
