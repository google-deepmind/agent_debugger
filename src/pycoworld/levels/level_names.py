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

"""Level names for backward compatibility.

This module lists the names of levels for backward compatibililty with old code.
We plan to deprecate this module in the future so please do not rely on it for
new code.
"""

from agent_debugger.src.pycoworld.levels import apples
from agent_debugger.src.pycoworld.levels import base_level
from agent_debugger.src.pycoworld.levels import color_memory
from agent_debugger.src.pycoworld.levels import grass_sand
from agent_debugger.src.pycoworld.levels import key_door
from agent_debugger.src.pycoworld.levels import red_green_apples


# Provided for backward compatibility only. Please do not use in new code.
def level_by_name(level_name: str) -> base_level.PycoworldLevel:
  """Returns the level corresponding to a given level name.

  Provided for backward compatibility only. Please do not use in new code.

  Args:
    level_name: Name of the level to construct. See NAMED_LEVELS for the
      supported level names.
  """
  if level_name == 'apples_corner':
    return apples.ApplesLevel(start_type='corner')
  elif level_name == 'apples_full':
    return apples.ApplesLevel()
  elif level_name == 'grass_sand':
    return grass_sand.GrassSandLevel()
  elif level_name == 'grass_sand_uncorrelated':
    return grass_sand.GrassSandLevel(corr=0.5, double_terminal_event=True)
  elif level_name == 'key_door':
    return key_door.KeyDoorLevel()
  elif level_name == 'key_door_closed':
    return key_door.KeyDoorLevel(door_type='closed')
  elif level_name == 'large_color_memory':
    return color_memory.ColorMemoryLevel(large=True)
  elif level_name == 'red_green_apples':
    return red_green_apples.RedGreenApplesLevel()

  raise ValueError(f'Unknown level {level_name}.')
