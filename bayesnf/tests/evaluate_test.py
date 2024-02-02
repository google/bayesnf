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

from bayesnf import dataset_config
from bayesnf import evaluate
import jax
import pandas as pd
import pytest


@pytest.fixture(name='golden_data')
def fixture_golden_data(fname):
  return pd.read_csv(io.BytesIO(pkgutil.get_data(
      'bayesnf', f'tests/test_data/{fname}')))


def run_objective(objective, inference_config):
  dataset = 'chickenpox'
  series_id = '8'
  output_dir = tempfile.mkdtemp()
  fname = f'bnf-{objective}.chickenpox.8.pred.csv'
  _ = evaluate.run_experiment(
      dataset=dataset,
      data_root='test_data/',
      series_id=series_id,
      output_dir=output_dir,
      objective=objective,
      dataset_config=dataset_config.DATASET_CONFIG[dataset],
      model_config=dataset_config.MODEL_CONFIG[dataset][objective],
      inference_config=inference_config,
      seed=jax.random.PRNGKey(0),
  )
  return pd.read_csv(os.path.join(output_dir, fname))


def test_map(golden_data):
  # These are from `runner.py`, but `num_epochs` is smaller for testing.
  # Note that `runner.py` is a binary, which is why this is not imported.
  inference_config = {
      'num_particles': 64,
      'num_epochs': 100,
      'learning_rate': 0.005}
  new_data = run_objective('map', inference_config)
  assert new_data.equals(golden_data)


def test_mle(golden_data):
  # These are from `runner.py`, but `num_epochs` is smaller for testing.
  # Note that `runner.py` is a binary, which is why this is not imported.
  inference_config = {
      'num_particles': 64,
      'num_epochs': 100,
      'learning_rate': 0.005}
  new_data = run_objective('mle', inference_config)
  assert new_data.equals(golden_data)


def test_vi(golden_data):
  # These are from `runner.py`, but `num_epochs` is smaller for testing.
  # Note that `runner.py` is a binary, which is why this is not imported.
  inference_config = {
      'batch_size': 511,
      'kl_weight': 0.1,
      'learning_rate': 0.01,
      'num_epochs': 100,
      'num_particles': 64,
      'sample_size': 5}
  new_data = run_objective('vi', inference_config)
  assert new_data.equals(golden_data)
