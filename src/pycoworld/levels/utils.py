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

"""Utilities for constructing pycoworld levels."""

from typing import Optional, Sequence, Tuple

import chex
import jax
import numpy as np

from agent_debugger.src.pycoworld import default_constants

Tile = default_constants.Tile

# A 2d slice for masking out the walls of a room.
_INTERIOR = (slice(1, -1),) * 2


def room(
    height: int,
    width: int,
    floor_tile: Tile = Tile.FLOOR,
    wall_tile: Tile = Tile.WALL,
) -> np.ndarray:
  """Returns a pycoworld representation of a room (floor surrounded by walls).

  Args:
    height: Height of the environment.
    width: Width of the environment.
    floor_tile: Tile to use for the floor.
    wall_tile: Tile to use for the wall.

  Returns:
    The array representation of the room.
  """
  room_array = np.full([height, width], wall_tile, dtype=np.uint8)
  room_array[_INTERIOR] = floor_tile
  return room_array


def sample_positions(
    rng: np.ndarray,
    height_range: Tuple[int, int],
    width_range: Tuple[int, int],
    number_samples: int,
    replace: bool = True,
    exclude: Optional[np.ndarray] = None,
) -> Sequence[Tuple[int, int]]:
  """Uniformly samples random positions in a 2d grid.

  Args:
    rng: Jax random key to use for sampling.
    height_range: Range (min, 1+max) in the height of the grid to sample from.
    width_range: Range (min, 1+max) in the width of the grid to sample from.
    number_samples: Number of positions to sample.
    replace: Whether to sample with replacement.
    exclude: Array of positions to exclude from the sampling. Each row of the
      array represents the x,y coordinates of one position.

  Returns:
    A sequence of x,y index tuples which can be used to index a 2d numpy array.

  Raises:
    ValueError: if more positions are requested than are available when sampling
      without replacement.
  """
  height = height_range[1] - height_range[0]
  width = width_range[1] - width_range[0]
  offset = np.array([height_range[0], width_range[0]])

  choices = np.arange(height * width)
  if exclude is not None:
    exclude = np.asarray(exclude)  # Allow the user to pass any array-like type.
    chex.assert_shape(exclude, (None, 2))

    exclude_offset = exclude - offset
    exclude_indices = np.ravel_multi_index(exclude_offset.T, (height, width))
    mask = np.ones_like(choices, dtype=bool)
    mask[exclude_indices] = False
    choices = choices[mask]

  flat_indices = jax.random.choice(
      rng, choices, (number_samples,), replace=replace)
  positions_offset = np.unravel_index(flat_indices, (height, width))
  positions = positions_offset + offset[:, np.newaxis]
  return list(zip(*positions))
