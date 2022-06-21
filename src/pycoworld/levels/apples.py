# Copyright 2022 DeepMind Technologies Limited
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

"""Apples level."""
from typing import Tuple

import numpy as np

from agent_debugger.src.pycoworld import default_constants
from agent_debugger.src.pycoworld.levels import base_level
from agent_debugger.src.pycoworld.levels import utils

Tile = default_constants.Tile


class ApplesLevel(base_level.PycoworldLevel):
  """Level where the goal is to pick up the reward."""

  def __init__(
      self,
      start_type: str = 'full_room',
      height: int = 8,
      width: int = 8,
  ) -> None:
    """Initializes the level.

    Args:
      start_type: Type of random initialization: 'full_room' forces random
        initialization using all cells in the room, 'corner' forces random
        initialization of the player around one corner of the room and the
        reward is initialized close to the other corner.
      height: Height of the grid.
      width: Width of the grid.
    """
    if start_type not in ['full_room', 'corner']:
      raise ValueError('Unrecognised start type.')
    self._start_type = start_type
    self._height = height
    self._width = width

  def foreground_and_background(
      self,
      rng: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """See base class."""
    foreground = utils.room(self._height, self._width)

    # Sample player's and reward's initial position in the interior of the room.
    if self._start_type == 'full_room':
      player_pos, reward_pos = utils.sample_positions(
          rng,
          height_range=(1, self._height - 1),
          width_range=(1, self._width - 1),
          number_samples=2,
          replace=False)
    elif self._start_type == 'corner':
      # Sample positions in the top left quadrant for the player
      player_pos = utils.sample_positions(
          rng,
          height_range=(1, self._height // 2),
          width_range=(1, self._width // 2),
          number_samples=1)[0]
      # Sample positions in the bottom right quadrant for the reward
      reward_pos = utils.sample_positions(
          rng,
          height_range=((self._height + 1) // 2, self._height - 1),
          width_range=((self._width + 1) // 2, self._width - 1),
          number_samples=1)[0]

    foreground[player_pos] = Tile.PLAYER
    foreground[reward_pos] = Tile.REWARD

    background = np.full_like(foreground, Tile.FLOOR)
    background[reward_pos] = Tile.TERMINAL_R
    return foreground, background
