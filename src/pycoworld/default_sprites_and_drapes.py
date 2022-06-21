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

"""This file contains the default Sprites and Drapes used in Pycoworld.

Sprites and Drapes are concepts part of Pycolab, the game engine we use. Sprites
are unique entities which can usually move and act in the environment. Drapes
are sets of entities which are not interacting with the environment, mostly
idle.
"""

import abc
from typing import MutableMapping, Any

import numpy as np
from pycolab import engine
from pycolab import things as plab_things

from agent_debugger.src.pycoworld import default_constants

Tile = default_constants.Tile


class PlayerSprite(plab_things.Sprite):
  """A `Sprite` for our player.

  The player can move around freely, as long as it doesn't attempt to
  move into an `IMPASSABLE` tile. There's also additional effects for
  certain special tiles, such as a goal, lava, and water tile.
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
    ypos, xpos = self.position
    height, width = board.shape

    # Where does the agent want to move?
    if actions == 0 and ypos >= 1:  # go upward?
      if board[ypos - 1, xpos] not in default_constants.IMPASSABLE:
        self._position = self.Position(ypos - 1, xpos)
    elif actions == 1 and ypos <= height - 2:  # go downward?
      if board[ypos + 1, xpos] not in default_constants.IMPASSABLE:
        self._position = self.Position(ypos + 1, xpos)
    elif actions == 2 and xpos >= 1:  # go leftward?
      if board[ypos, xpos - 1] not in default_constants.IMPASSABLE:
        self._position = self.Position(ypos, xpos - 1)
    elif actions == 3 and xpos <= width - 2:  # go rightward?
      if board[ypos, xpos + 1] not in default_constants.IMPASSABLE:
        self._position = self.Position(ypos, xpos + 1)
    ypos, xpos = self.position

    floor_tile = backdrop.curtain[ypos, xpos]
    terminals = [
        Tile.TERMINAL, Tile.TERMINAL_R, Tile.TERMINAL_G, Tile.TERMINAL_B
    ]
    if floor_tile in terminals:
      the_plot.terminate_episode()

    if floor_tile in [Tile.LAVA]:
      the_plot.add_reward(default_constants.REWARD_LAVA)
      the_plot.terminate_episode()

    if floor_tile in [Tile.WATER]:
      the_plot.add_reward(default_constants.REWARD_WATER)


class RewardDrape(plab_things.Drape):
  """A drape for small rewards."""

  def update(
      self,
      actions: int,
      board: np.ndarray,
      layers: MutableMapping[str, np.ndarray],
      backdrop: np.ndarray,
      things: MutableMapping[str, Any],
      the_plot: engine.plot.Plot
  ) -> None:
    ypos, xpos = things[chr(Tile.PLAYER)].position
    if self.curtain[ypos, xpos]:
      the_plot.add_reward(default_constants.REWARD_SMALL)
      self.curtain[ypos, xpos] = False


class BigRewardDrape(plab_things.Drape):
  """A drape for big rewards."""

  def update(
      self,
      actions: int,
      board: np.ndarray,
      layers: MutableMapping[str, np.ndarray],
      backdrop: np.ndarray,
      things: MutableMapping[str, Any],
      the_plot: engine.plot.Plot
  ) -> None:
    ypos, xpos = things[chr(Tile.PLAYER)].position
    if self.curtain[ypos, xpos]:
      the_plot.add_reward(default_constants.REWARD_BIG)
      self.curtain[ypos, xpos] = False


class ObjectDrape(plab_things.Drape):
  """A drape with objects that the agent can carry in its inventory."""

  def update(
      self,
      actions: int,
      board: np.ndarray,
      layers: MutableMapping[str, np.ndarray],
      backdrop: np.ndarray,
      things: MutableMapping[str, Any],
      the_plot: engine.plot.Plot
  ) -> None:
    ypos, xpos = things[chr(Tile.PLAYER)].position

    if self.character not in the_plot.keys():
      # The inventory is empty by default.
      the_plot[self.character] = 0

    if self.curtain[ypos, xpos]:
      the_plot[self.character] = the_plot[self.character] + 1
      self.curtain[ypos, xpos] = False


class WallDrape(plab_things.Drape):
  """A drape for walls, which does nothing."""

  def update(
      self,
      actions: int,
      board: np.ndarray,
      layers: MutableMapping[str, Any],
      backdrop: np.ndarray,
      things: MutableMapping[str, Any],
      the_plot: engine.plot.Plot
  ) -> None:
    """Updates the environment with the actions."""


class SensorDrape(plab_things.Drape, abc.ABC):
  """A drape for plain sensors, triggering some function in the environment."""

  @abc.abstractmethod
  def _trigger(
      self,
      board: np.ndarray,
      layers: MutableMapping[str, np.ndarray],
      backdrop: np.ndarray,
      things: MutableMapping[str, Any],
      the_plot: engine.plot.Plot
  ) -> None:
    """Triggers something in the environment as soon as the sensor is pushed."""

  def update(
      self,
      actions: int,
      board: np.ndarray,
      layers: MutableMapping[str, np.ndarray],
      backdrop: np.ndarray,
      things: MutableMapping[str, Any],
      the_plot: engine.plot.Plot
  ) -> None:
    ypos, xpos = things[chr(default_constants.Tile.PLAYER)].position
    if self.curtain[ypos, xpos]:
      # As soon as the agent steps on the sensor, we trigger it.
      self._trigger(board, layers, backdrop, things, the_plot)


class DoorDrape(plab_things.Drape):
  """A drape with doors tiles.

  Doors come in different colors. They act as impassable tiles, unless the
  player has a key of the associated color in its inventory: in this case,
  the key is consumed and the door disappears.
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
    ypos, xpos = things[chr(Tile.PLAYER)].position
    height, width = self.curtain.shape

    # What's the required key?
    key = chr(ord(self.character) - 4)

    # Where does the agent want to move?
    # If the agent wants to move into a door, then check whether
    # the corresponding key is available. If it is, then remove the door.
    if actions == 0 and ypos >= 1:  # go upward?
      if self.curtain[ypos - 1, xpos] and the_plot[key] > 0:
        self.curtain[ypos - 1, xpos] = False
        the_plot[key] = the_plot[key] - 1
    elif actions == 1 and ypos <= height - 2:  # go downward?
      if self.curtain[ypos + 1, xpos] and the_plot[key] > 0:
        self.curtain[ypos + 1, xpos] = False
        the_plot[key] = the_plot[key] - 1
    elif actions == 2 and xpos >= 1:  # go leftward?
      if self.curtain[ypos, xpos - 1] and the_plot[key] > 0:
        self.curtain[ypos, xpos - 1] = False
        the_plot[key] = the_plot[key] - 1
    elif actions == 3 and xpos <= width - 2:  # go rightward?
      if self.curtain[ypos, xpos + 1] and the_plot[key] > 0:
        self.curtain[ypos, xpos + 1] = False
        the_plot[key] = the_plot[key] - 1


class BoxDrape(plab_things.Drape):
  """A drape for pushable blocks.

  These blocks can be pushed by the player if there is a free space behind.
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
    ypos, xpos = things[chr(Tile.PLAYER)].position
    height, width = self.curtain.shape

    # Where does the agent want to move?
    if actions == 0 and ypos >= 2:  # go upward?
      check_impassable = board[ypos - 2,
                               xpos] not in default_constants.IMPASSABLE
      if self.curtain[ypos - 1, xpos] and check_impassable:
        self.curtain[ypos - 1, xpos] = False
        self.curtain[ypos - 2, xpos] = True
    elif actions == 1 and ypos <= height - 3:  # go downward?
      check_impassable = board[ypos + 2,
                               xpos] not in default_constants.IMPASSABLE
      if self.curtain[ypos + 1, xpos] and check_impassable:
        self.curtain[ypos + 1, xpos] = False
        self.curtain[ypos + 2, xpos] = True
    elif actions == 2 and xpos >= 2:  # go leftward?
      check_impassable = board[ypos,
                               xpos - 2] not in default_constants.IMPASSABLE
      if self.curtain[ypos, xpos - 1] and check_impassable:
        self.curtain[ypos, xpos - 1] = False
        self.curtain[ypos, xpos - 2] = True
    elif actions == 3 and xpos <= width - 3:  # go rightward?
      check_impassable = board[ypos,
                               xpos + 2] not in default_constants.IMPASSABLE
      if self.curtain[ypos, xpos + 1] and check_impassable:
        self.curtain[ypos, xpos + 1] = False
        self.curtain[ypos, xpos + 2] = True
