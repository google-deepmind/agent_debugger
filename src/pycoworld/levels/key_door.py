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

"""Key Door level.

In this level there is a key and a door. Behind the door there is a reward. The
agent must learn to collect the key, open the door and obtain the reward.
"""

import jax
import jax.numpy as jnp
import numpy as np

from agent_debugger.src.pycoworld import default_constants
from agent_debugger.src.pycoworld.levels import base_level
from agent_debugger.src.pycoworld.levels import utils

_VALID_DOORS = ['random', 'closed', 'open']


class KeyDoorLevel(base_level.PycoworldLevel):
  """Level in which the agent has to unlock a door to collect the reward."""

  def __init__(self, door_type: str = 'random') -> None:
    """Initializes the level.

    Args:
      door_type: The way we sample the door state (open or closed). Must be in
        {'random', 'closed', 'open'}, otherwise a ValueError is raised.
    """
    if door_type not in _VALID_DOORS:
      raise ValueError(
          f'Argument door_type has an incorrect value. Expected one '
          f'of {_VALID_DOORS}, but got {door_type} instead.')
    self._door_type = door_type

  def foreground_and_background(
      self,
      rng: jnp.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns a tuple with the level foreground and background.

    We first determine the state of the door and use the tile associated with
    it (closed and open doors don't have the same id).
    Then, we sample the random player and key positions and add their tiles to
    the board. The background is full of floor tiles and a terminal event
    beneath the reward tile, so that the episode is terminated when the agent
    gets the reward.

    Args:
      rng: The jax random seed. Standard name for jax random seeds.
    """
    if self._door_type == 'closed':
      door_state = 'closed'
    elif self._door_type == 'open':
      door_state = 'open'
    elif self._door_type == 'random':
      rng, rng1 = jax.random.split(rng)
      if jax.random.uniform(rng1) < 0.5:
        door_state = 'open'
      else:
        door_state = 'closed'

    foreground = np.array([[4, 4, 4, 4, 4, 0, 0], [4, 0, 0, 0, 4, 0, 0],
                           [4, 0, 0, 0, 4, 4, 4], [4, 0, 0, 0, 0, 20, 4],
                           [4, 4, 4, 4, 4, 4, 4]])

    if door_state == 'closed':
      foreground[3, 4] = default_constants.Tile.DOOR_R

    player_pos, key_pos = utils.sample_positions(
        rng,
        height_range=(1, 4),
        width_range=(1, 4),
        number_samples=2,
        replace=False)

    foreground[player_pos] = default_constants.Tile.PLAYER
    foreground[key_pos] = default_constants.Tile.KEY_R

    background = np.zeros(foreground.shape, dtype=int)
    background[3, 5] = default_constants.Tile.TERMINAL_R
    return foreground, background
