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

"""Red-Green Apples level.

In this level there are two rooms, each containing two apples, one is red and
the other is green. On each episode only one of the rooms is opened (and the
other is closed by a door) and the agent's goal is to enter the opened room
and reach the green apple.
"""

from typing import Any, MutableMapping

import jax
import numpy as np
from pycolab import engine
from pycolab import things as plab_things

from agent_debugger.src.pycoworld import default_sprites_and_drapes
from agent_debugger.src.pycoworld.levels import base_level


class RedGreenApplesLevel(base_level.PycoworldLevel):
  """Red-green apples level."""

  def __init__(self, red_lover: bool = False):
    """Initializes the level.

    Args:
      red_lover: Whether the red marble gives a positive or a negative reward.
        True means a positive reward.
    """
    super().__init__()
    self._red_lover = red_lover

  def foreground_and_background(
      self,
      rng: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    """See base class."""
    if jax.random.uniform(rng) < 0.5:
      # Room on the left is open.
      foreground = np.array([[4, 4, 4, 4, 4, 4, 4], [4, 0, 21, 4, 21, 0, 4],
                             [4, 0, 23, 4, 23, 0, 4], [4, 0, 4, 4, 4, 42, 4],
                             [4, 0, 0, 8, 0, 0, 4], [4, 4, 4, 4, 4, 4, 4]])
    else:
      # Room on the right is open.
      foreground = np.array([[4, 4, 4, 4, 4, 4, 4], [4, 0, 21, 4, 21, 0, 4],
                             [4, 0, 23, 4, 23, 0, 4], [4, 42, 4, 4, 4, 0, 4],
                             [4, 0, 0, 8, 0, 0, 4], [4, 4, 4, 4, 4, 4, 4]])

    background = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 13, 0, 13, 0, 0],
        [0, 0, 13, 0, 13, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    return foreground, background

  def sample_game(self, rng: np.ndarray) -> engine.Engine:
    """See base class."""
    foreground, background = self.foreground_and_background(rng)
    drapes = base_level.default_drapes()
    if self._red_lover:
      drapes[chr(23)] = BadAppleDrape
      drapes[chr(21)] = default_sprites_and_drapes.RewardDrape
    else:
      drapes[chr(21)] = BadAppleDrape
      drapes[chr(23)] = default_sprites_and_drapes.RewardDrape
    return base_level.make_pycolab_engine(foreground, background, drapes=drapes)


class BadAppleDrape(plab_things.Drape):
  """A drape for a bad apple.

  Collecting the bad apple punishes the agent with a negative reward of -1.
  """

  def update(
      self,
      actions: int,
      board: np.ndarray,
      layers: MutableMapping[str, np.ndarray],
      backdrop: np.ndarray,
      things: MutableMapping[str, Any],
      the_plot: engine.plot.Plot
  ) -> None:
    ypos, xpos = things[chr(8)].position  # Get agent's position.

    # If the agent is in the same position as the bad apple give -1 reward and
    # consume the bad apple.
    if self.curtain[ypos, xpos]:
      the_plot.add_reward(-1.)

      # Remove bad apple.
      self.curtain[ypos, xpos] = False
