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

"""Default constants for Pycoworld."""

import enum


class Tile(enum.IntEnum):
  """Available pycoworld tiles with their IDs.

  This IntEnum is only provided for better readability. There is no guarantee
  that the raw IDs are not used anywhere in the code and no requirement to use
  the enum in cases where the raw IDs are more readable.
  """
  FLOOR = 0
  FLOOR_R = 1
  FLOOR_B = 2
  FLOOR_G = 3

  WALL = 4
  WALL_R = 5
  WALL_B = 6
  WALL_G = 7

  PLAYER = 8
  PLAYER_R = 9
  PLAYER_B = 10
  PLAYER_G = 11

  TERMINAL = 12
  TERMINAL_R = 13
  TERMINAL_B = 14
  TERMINAL_G = 15

  BLOCK = 16
  BLOCK_R = 17
  BLOCK_B = 18
  BLOCK_G = 19

  REWARD = 20
  REWARD_R = 21
  REWARD_B = 22
  REWARD_G = 23

  BIG_REWARD = 24
  BIG_REWARD_R = 25
  BIG_REWARD_B = 26
  BIG_REWARD_G = 27

  HOLE = 28
  HOLE_R = 29
  HOLE_B = 30
  HOLE_G = 31

  SAND = 32
  GRASS = 33
  LAVA = 34
  WATER = 35

  KEY = 36
  KEY_R = 37
  KEY_B = 38
  KEY_G = 39

  DOOR = 40
  DOOR_R = 41
  DOOR_B = 42
  DOOR_G = 43

  SENSOR = 44
  SENSOR_R = 45
  SENSOR_B = 46
  SENSOR_G = 47

  OBJECT = 48
  OBJECT_R = 49
  OBJECT_B = 50
  OBJECT_G = 51


# This dict defines which sensor matches which object.
MATCHING_OBJECT_DICT = {
    Tile.SENSOR: Tile.OBJECT,
    Tile.SENSOR_R: Tile.OBJECT_R,
    Tile.SENSOR_G: Tile.OBJECT_G,
    Tile.SENSOR_B: Tile.OBJECT_B,
}

# Impassable tiles: these tiles cannot be traversed by the agent.
IMPASSABLE = [
    Tile.WALL,
    Tile.WALL_R,
    Tile.WALL_B,
    Tile.WALL_G,
    Tile.BLOCK,
    Tile.BLOCK_R,
    Tile.BLOCK_B,
    Tile.BLOCK_G,
    Tile.DOOR,
    Tile.DOOR_R,
    Tile.DOOR_B,
    Tile.DOOR_G,
]

REWARD_LAVA = -10.
REWARD_WATER = -1.
REWARD_SMALL = 1.
REWARD_BIG = 5.
