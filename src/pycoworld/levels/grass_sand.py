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

"""Grass sand level."""
from typing import Tuple, Optional

import jax
import numpy as np

from agent_debugger.src.pycoworld import default_constants
from agent_debugger.src.pycoworld.levels import base_level

Tile = default_constants.Tile


class GrassSandLevel(base_level.PycoworldLevel):
  """The grass sand level.

  The goal position causally depends on the type of world. Similarly, the floor
  tiles (grass or sand) also causally depend on the type of world. Therefore,
  type of floor and reward position are correlated, but there is no causal link
  between them.
  """

  def __init__(
      self,
      corr: float = 1.0,
      above: Optional[np.ndarray] = None,
      below: Optional[np.ndarray] = None,
      reward_pos: Optional[Tuple[np.ndarray, np.ndarray]] = None,
      double_terminal_event: Optional[bool] = False,
  ) -> None:
    """Initializes the level.

    Args:
      corr: "correlation" between reward position and world. 1.0 means that the
        reward position will always be on "north" for sand world and on "south"
        for grass world. A value of 0.5 corresponds to fully randomized position
        independent of the world.
      above: a map to use.
      below: a background to use.
      reward_pos: position of the rewards.
      double_terminal_event: whether to have a terminal event on both sides of
        the maze or only beneath the reward.
    """
    if not 0.0 <= corr <= 1.0:
      raise ValueError('corr variable must be a float between 0.0 and 1.0')

    self._corr = corr
    if above is None:
      above = np.array([[0, 0, 0, 4, 4, 4], [4, 4, 4, 4, 99, 4],
                        [4, 4, 4, 4, 99, 4], [4, 8, 99, 99, 99, 4],
                        [4, 4, 4, 4, 99, 4], [4, 4, 4, 4, 99, 4],
                        [0, 0, 0, 4, 4, 4]])
    self._above = above
    if below is None:
      below = np.full_like(above, Tile.FLOOR)
    self._below = below

    if reward_pos is None:
      self._reward_pos_top = np.array([1, 4])
      self._reward_pos_bottom = np.array([5, 4])
    else:
      self._reward_pos_top, self._reward_pos_bottom = reward_pos

    self._double_terminal_event = double_terminal_event

  def foreground_and_background(
      self,
      rng: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """See base class."""
    rng1, rng2 = jax.random.split(rng, 2)
    # Do not modify the base maps during sampling
    sampled_above = self._above.copy()
    sampled_below = self._below.copy()

    # Select world.
    if jax.random.uniform(rng1) < 0.5:
      world = 'sand'
      floor_type = Tile.SAND
    else:
      world = 'grass'
      floor_type = Tile.GRASS

    # Sample reward location depending on corr variable.
    if jax.random.uniform(rng2) <= self._corr:
      # Standard reward position for both worlds.
      if world == 'sand':
        used_reward_pos = self._reward_pos_top
      elif world == 'grass':
        used_reward_pos = self._reward_pos_bottom
    else:
      # Alternative reward position for both worlds.
      if world == 'sand':
        used_reward_pos = self._reward_pos_bottom
      elif world == 'grass':
        used_reward_pos = self._reward_pos_top

    if self._double_terminal_event:
      # Put terminal events everywhere first.
      sampled_above[self._reward_pos_top[0],
                    self._reward_pos_top[1]] = Tile.TERMINAL_R
      sampled_above[self._reward_pos_bottom[0],
                    self._reward_pos_bottom[1]] = Tile.TERMINAL_R
    # Draw the reward ball.
    sampled_above[used_reward_pos[0], used_reward_pos[1]] = Tile.REWARD

    # Substitute tiles with code 99 with the corresponding floor_type code.
    floor_tiles = np.argwhere(sampled_above == 99)
    for tile in floor_tiles:
      sampled_above[tile[0], tile[1]] = floor_type

    if self._double_terminal_event:
      # Add terminal events on both sides to the below.
      sampled_below[self._reward_pos_top[0],
                    self._reward_pos_top[1]] = Tile.TERMINAL_R
      sampled_below[self._reward_pos_bottom[0],
                    self._reward_pos_bottom[1]] = Tile.TERMINAL_R
    else:
      sampled_below[used_reward_pos[0], used_reward_pos[1]] = Tile.TERMINAL_R

    return sampled_above, sampled_below
