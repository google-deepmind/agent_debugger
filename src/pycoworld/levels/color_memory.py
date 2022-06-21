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

"""Color memory level."""

import numpy as np

from agent_debugger.src.pycoworld.levels import grass_sand


class ColorMemoryLevel(grass_sand.GrassSandLevel):
  """A modification of the grass sand environment to test memory.

  The position of the reward is correlated with the floor type, as
  in grass_sand, but the agent has an egocentric view. Therefore, it must
  remember the color at the beginning to get the reward as fast as possible.
  """

  def __init__(self, corr: float = 1.0, large: bool = False) -> None:
    """Initializes the level.

    Args:
      corr: see grass_sand
      large: whether to enlarge the initial corridor.
    """
    if not large:
      above = np.array([[0, 0, 0, 0, 4, 4, 4], [0, 0, 0, 0, 4, 0, 4],
                        [4, 4, 4, 4, 4, 0, 4], [4, 4, 4, 4, 4, 0, 4],
                        [4, 99, 8, 0, 0, 0, 4], [4, 4, 4, 4, 4, 0, 4],
                        [4, 4, 4, 4, 4, 0, 4], [0, 0, 0, 0, 4, 0, 4],
                        [0, 0, 0, 0, 4, 4, 4]])
    else:
      above = np.array([[0, 0, 0, 0, 4, 4, 4], [0, 0, 0, 0, 4, 0, 4],
                        [4, 4, 4, 4, 4, 0, 4], [4, 4, 0, 0, 0, 0, 4],
                        [4, 99, 8, 0, 0, 0, 4], [4, 4, 0, 0, 0, 0, 4],
                        [4, 4, 4, 4, 4, 0, 4], [0, 0, 0, 0, 4, 0, 4],
                        [0, 0, 0, 0, 4, 4, 4]])
    below = np.zeros(above.shape, dtype=np.uint8)
    reward_pos = (np.array([1, 5]), np.array([7, 5]))
    super().__init__(corr, above, below, reward_pos, double_terminal_event=True)
