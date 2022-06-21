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

"""Level config."""

import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pycolab import ascii_art
from pycolab import engine
from pycolab import things as pycolab_things

from agent_debugger.src.pycoworld import default_constants
from agent_debugger.src.pycoworld import default_sprites_and_drapes

_ACTIONS = range(4)
Tile = default_constants.Tile


def default_sprites() -> Dict[str, Any]:
  """Returns the default mapping of characters to sprites in the levels."""
  return {chr(Tile.PLAYER): default_sprites_and_drapes.PlayerSprite}


def default_drapes() -> Dict[str, Any]:
  """Returns the default mapping of characters to drapes in the levels."""
  return {
      chr(Tile.WALL): default_sprites_and_drapes.WallDrape,
      chr(Tile.WALL_R): default_sprites_and_drapes.WallDrape,
      chr(Tile.WALL_B): default_sprites_and_drapes.WallDrape,
      chr(Tile.WALL_G): default_sprites_and_drapes.WallDrape,
      chr(Tile.BLOCK): default_sprites_and_drapes.BoxDrape,
      chr(Tile.BLOCK_R): default_sprites_and_drapes.BoxDrape,
      chr(Tile.BLOCK_B): default_sprites_and_drapes.BoxDrape,
      chr(Tile.BLOCK_G): default_sprites_and_drapes.BoxDrape,
      chr(Tile.REWARD): default_sprites_and_drapes.RewardDrape,
      chr(Tile.REWARD_R): default_sprites_and_drapes.RewardDrape,
      chr(Tile.REWARD_B): default_sprites_and_drapes.RewardDrape,
      chr(Tile.REWARD_G): default_sprites_and_drapes.RewardDrape,
      chr(Tile.BIG_REWARD): default_sprites_and_drapes.BigRewardDrape,
      chr(Tile.BIG_REWARD_R): default_sprites_and_drapes.BigRewardDrape,
      chr(Tile.BIG_REWARD_B): default_sprites_and_drapes.BigRewardDrape,
      chr(Tile.BIG_REWARD_G): default_sprites_and_drapes.BigRewardDrape,
      chr(Tile.KEY): default_sprites_and_drapes.ObjectDrape,
      chr(Tile.KEY_R): default_sprites_and_drapes.ObjectDrape,
      chr(Tile.KEY_B): default_sprites_and_drapes.ObjectDrape,
      chr(Tile.KEY_G): default_sprites_and_drapes.ObjectDrape,
      chr(Tile.DOOR): default_sprites_and_drapes.DoorDrape,
      chr(Tile.DOOR_R): default_sprites_and_drapes.DoorDrape,
      chr(Tile.DOOR_B): default_sprites_and_drapes.DoorDrape,
      chr(Tile.DOOR_G): default_sprites_and_drapes.DoorDrape,
      chr(Tile.OBJECT): default_sprites_and_drapes.ObjectDrape,
      chr(Tile.OBJECT_R): default_sprites_and_drapes.ObjectDrape,
      chr(Tile.OBJECT_B): default_sprites_and_drapes.ObjectDrape,
      chr(Tile.OBJECT_G): default_sprites_and_drapes.ObjectDrape,
  }


def default_schedule() -> List[str]:
  """Returns the default update schedule of sprites and drapes in the levels."""
  return [
      chr(Tile.PLAYER),  # PlayerSprite
      chr(Tile.WALL),
      chr(Tile.WALL_R),
      chr(Tile.WALL_B),
      chr(Tile.WALL_G),
      chr(Tile.REWARD),
      chr(Tile.REWARD_R),
      chr(Tile.REWARD_B),
      chr(Tile.REWARD_G),
      chr(Tile.BIG_REWARD),
      chr(Tile.BIG_REWARD_R),
      chr(Tile.BIG_REWARD_B),
      chr(Tile.BIG_REWARD_G),
      chr(Tile.KEY),
      chr(Tile.KEY_R),
      chr(Tile.KEY_B),
      chr(Tile.KEY_G),
      chr(Tile.OBJECT),
      chr(Tile.OBJECT_R),
      chr(Tile.OBJECT_B),
      chr(Tile.OBJECT_G),
      chr(Tile.DOOR),
      chr(Tile.DOOR_R),
      chr(Tile.DOOR_B),
      chr(Tile.DOOR_G),
      chr(Tile.BLOCK),
      chr(Tile.BLOCK_R),
      chr(Tile.BLOCK_B),
      chr(Tile.BLOCK_G),
  ]


def _numpy_to_str(array: np.ndarray) -> List[str]:
  """Converts numpy array into a list of strings.

  Args:
    array: a 2-D np.darray of np.uint8.

  Returns:
    A list of strings of equal length, corresponding to the entries in A.
  """
  return [''.join(map(chr, row.tolist())) for row in array]


def make_pycolab_engine(
    foreground: np.ndarray,
    background: Union[np.ndarray, int],
    sprites: Optional[Mapping[str, pycolab_things.Sprite]] = None,
    drapes: Optional[Mapping[str, pycolab_things.Drape]] = None,
    update_schedule: Optional[Sequence[str]] = None,
    rng: Optional[Any] = None,
) -> engine.Engine:
  """Builds and returns a pycoworld game engine.

  Args:
    foreground: Array of foreground tiles.
    background: Array of background tiles or a single tile to use as the
      background everywhere.
    sprites: Pycolab sprites. See pycolab.ascii_art.ascii_art_to_game for more
      information.
    drapes: Pycolab drapes.
    update_schedule: Update schedule for sprites and drapes.
    rng: Random key to use for pycolab.

  Returns:
    A pycolab engine with the pycoworld game.
  """
  sprites = sprites if sprites is not None else default_sprites()
  drapes = drapes if drapes is not None else default_drapes()
  if update_schedule is None:
    update_schedule = default_schedule()

  # The pycolab engine constructor requires arrays of strings
  above_str = _numpy_to_str(foreground)
  below_str = _numpy_to_str(background)

  pycolab_engine = ascii_art.ascii_art_to_game(
      above_str, below_str, sprites, drapes, update_schedule=update_schedule)
  # Pycolab does not allow to add a global seed in the engine constructor.
  # Therefore, we have to set it manually.
  pycolab_engine._the_plot['rng'] = rng  # pylint: disable=protected-access
  return pycolab_engine


class PycoworldLevel(abc.ABC):
  """Abstract class representing a pycoworld level.

  A pycoworld level captures all the data that is required to define the level
  and implements a function that returns a pycolab game engine.
  """

  @abc.abstractmethod
  def foreground_and_background(
      self,
      rng: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the foreground and background arrays of the level."""

  def sample_game(self, rng: np.ndarray) -> engine.Engine:
    """Samples and returns a game from this level.

    A level may contain random elements (e.g. a random goal location). This
    function samples and returns a particular game from this level. The base
    version calls foreground_and_background and constructs a pycolab engine
    using the arrays. In order to use custom sprites, drapes, and update
    schedule override this method and use make_pycolab_engine to create an
    engine.

    Args:
      rng: Random key to use for sampling.

    Returns:
      A pycolab game engine for the sampled game.
    """
    foreground, background = self.foreground_and_background(rng)
    return make_pycolab_engine(foreground, background)

  @property
  def actions(self) -> Sequence[int]:
    return _ACTIONS
