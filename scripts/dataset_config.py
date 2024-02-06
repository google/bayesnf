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

"""Dataset configuration in spatiotemporal experiments."""

import numpy

DATASET_CONFIG = {
    'air_quality': {
        'num_series': 12,
        'target_col': 'pm10',
        'timetype': 'index',
        'freq': 'H',
        'feature_cols': ['datetime', 'latitude', 'longitude'],
        'standardize': ['latitude', 'longitude'],
        'series_id_fmt': str,
    },
    'wind': {
        'num_series': 12,
        'target_col': 'wind',
        'timetype': 'index',
        'freq': 'D',
        'feature_cols': ['datetime', 'latitude', 'longitude'],
        'standardize': ['latitude', 'longitude'],
        'series_id_fmt': str,
    },
    'air': {
        'num_series': 12,
        'target_col': 'pm10',
        'timetype': 'index',
        'freq': 'D',
        'feature_cols': ['datetime', 'latitude', 'longitude'],
        'standardize': ['latitude', 'longitude'],
        'series_id_fmt': str,
    },
    'chickenpox': {
        'num_series': 12,
        'target_col': 'chickenpox',
        'timetype': 'index',
        'freq': 'W',
        'feature_cols': ['datetime', 'latitude', 'longitude'],
        'standardize': ['latitude', 'longitude'],
        'series_id_fmt': str,
    },
    'coprecip': {
        'num_series': 12,
        'target_col': 'ppt',
        'timetype': 'index',
        'freq': 'M',
        'feature_cols': ['datetime', 'latitude', 'longitude'],
        'standardize': ['latitude', 'longitude'],
        'series_id_fmt': str,
    },
    'sst': {
        'num_series': 12,
        'target_col': 'sst',
        'timetype': 'index',
        'freq': 'M',
        'feature_cols': ['datetime', 'latitude', 'longitude', 'soi'],
        'standardize': ['latitude', 'longitude'],
        'series_id_fmt': str,
    },
    'M3Month': {
        'num_series': 1428,
        'target_col': 'value',
        'timetype': 'index',
        'freq': 'M',
        'feature_cols': ['datetime'],
        'standardize': [],
        'series_id_fmt': lambda s: str(1402 + s),
    },
}


def _get_model_config():
  """Return model configs."""
  ret = {}

  ret['air_quality'] = {
      'map': {
          'width': 512,
          'depth': 2,
          'seasonality_periods': numpy.asarray([24, 24 * 7]),
          'num_seasonal_harmonics': numpy.asarray([4, 4]),
          'observation_model': 'NORMAL',
      }
  }
  ret['air_quality']['mle'] = ret['air_quality']['map']
  ret['air_quality']['vi'] = ret['air_quality']['map'] | {
      'width': 512,
      'observation_model': 'NORMAL',
  }

  ret['wind'] = {
      'map': {
          'width': 512,
          'depth': 2,
          'seasonality_periods': numpy.asarray([7, 365.25 / 12, 365.25]),
          'num_seasonal_harmonics': numpy.asarray([3, 10, 10]),
          'observation_model': 'NORMAL',
      }
  }
  ret['wind']['mle'] = ret['wind']['map']
  ret['wind']['vi'] = ret['wind']['map'] | {
      'observation_model': 'NORMAL',
  }

  ret['air'] = {
      'map': {
          'width': 512,
          'depth': 2,
          'seasonality_periods': numpy.asarray([7, 365.25 / 12, 365.25]),
          'num_seasonal_harmonics': numpy.asarray([3, 10, 10]),
          'observation_model': 'NORMAL',
      }
  }
  ret['air']['mle'] = ret['air']['map']
  ret['air']['vi'] = ret['air']['map'] | {
      'depth': 2,
      'observation_model': 'NORMAL',
  }

  ret['chickenpox'] = {
      'map': {
          'width': 256,
          'depth': 2,
          'seasonality_periods': numpy.asarray([
              4.0,
              52.1775,
          ]),
          'num_seasonal_harmonics': numpy.asarray([2.0, 10]),
          'observation_model': 'NORMAL',
      }
  }
  ret['chickenpox']['mle'] = ret['chickenpox']['map']
  ret['chickenpox']['vi'] = ret['chickenpox']['map'] | {
      'observation_model': 'NORMAL',
  }

  ret['coprecip'] = {
      'map': {
          'width': 512,
          'depth': 2,
          'seasonality_periods': numpy.asarray([12]),
          'num_seasonal_harmonics': numpy.asarray([6]),
          'observation_model': 'NORMAL',
      }
  }
  ret['coprecip']['mle'] = ret['coprecip']['map']
  ret['coprecip']['vi'] = ret['coprecip']['map']

  ret['sst'] = {
      'map': {
          'width': 768,
          'depth': 2,
          'seasonality_periods': numpy.asarray([
              12,
          ]),
          'num_seasonal_harmonics': numpy.asarray([
              6,
          ]),
          'observation_model': 'NORMAL',
      }
  }
  ret['sst']['mle'] = ret['sst']['map']
  ret['sst']['vi'] = ret['sst']['map']

  ret['M3Month'] = {
      'map': {
          'width': 1024,
          'depth': 2,
          'seasonality_periods': numpy.asarray([12]),
          'num_seasonal_harmonics': numpy.asarray([6]),
      }
  }
  ret['M3Month']['mle'] = ret['M3Month']['map']

  return ret


MODEL_CONFIG = _get_model_config()
