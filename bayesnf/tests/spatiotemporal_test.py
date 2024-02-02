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

"""Tests for spatiotemporal.py."""
from bayesnf import spatiotemporal
import numpy as np
import pytest


@pytest.mark.parametrize(
    "seasonality, freq, expected",
    [
        ("Y", "Y", 1),
        ("Q", "Q", 1),
        ("Y", "Q", 4),
        ("M", "H", 730.5),
        ("Q", "M", 3),
        ("Y", "M", 12),
        ("M", "D", 30.4375),
        ("min", "S", 60),
        ("H", "S", 3600),
        ("D", "S", 86400),
        ("M", "S", 2629800),
        ("Q", "S", 7889400),
        ("Y", "S", 31557600),
    ],
)
def test_seasonality_to_float(seasonality, freq, expected):
  assert spatiotemporal.seasonality_to_float(seasonality, freq) == expected


def test_seasonalities_to_array():
  periods = spatiotemporal.seasonalities_to_array(["D", "W", "M"], "H")
  np.testing.assert_allclose(periods, np.array([24, 168, 730.5]))
