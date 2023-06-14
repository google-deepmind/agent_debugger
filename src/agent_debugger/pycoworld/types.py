# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Specific types for the Pycoworld debugger."""

from typing import Tuple, Union

from pycolab import engine as engine_lib
from pycolab import things

PycolabPosition = things.Sprite.Position
Position = Union[PycolabPosition, Tuple[float, float]]
PycolabEngine = engine_lib.Engine


def to_pycolab_position(pos: Position) -> PycolabPosition:
  """Returns a PycolabPosition from a Tuple or a PycolabPosition."""
  if isinstance(pos, tuple):
    return PycolabPosition(*pos)
  return pos
