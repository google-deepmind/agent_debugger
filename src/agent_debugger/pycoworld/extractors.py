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

"""Pycoworld extractors.

The extractors act on the node directly, and extract information from it.
"""

import numpy as np
from pycolab import things

from agent_debugger.src.agent_debugger import node as node_lib
from agent_debugger.src.agent_debugger.pycoworld import types


class PycoworldExtractors:
  """Object containing the pycoworld extractors."""

  # pylint: disable=protected-access
  def get_element_positions(
      self,
      node: node_lib.Node,
      element_id: int,
  ) -> list[types.Position]:
    """Returns the position of a sprite/drape/background element.

    Args:
      node: The node to intervene on.
      element_id: The identifier of the element. A full list can be found in
        the pycoworld/default_constants.py file.
    """
    engine = node.env_state.current_game
    element_key = chr(element_id)
    if element_key in engine._sprites_and_drapes:
      # If it is a sprite.
      if isinstance(engine._sprites_and_drapes[element_key], things.Sprite):
        return [engine._sprites_and_drapes[element_key].position]
      # Otherwise, it's a drape.
      else:
        curtain = engine._sprites_and_drapes[element_key].curtain
        list_tuples = list(zip(*np.where(curtain)))
        return [types.PycolabPosition(*x) for x in list_tuples]

    # Last resort: we look at the board and the backdrop.
    board = engine._board.board
    list_tuples = list(zip(*np.where(board == ord(element_key))))
    backdrop = engine._backdrop.curtain
    list_tuples += list(zip(*np.where(backdrop == ord(element_key))))
    return [types.PycolabPosition(*x) for x in list_tuples]

  def get_backdrop_curtain(self, node: node_lib.Node) -> np.ndarray:
    """Returns the backdrop of a pycolab engine from a node."""
    return node.env_state.current_game._backdrop.curtain

  # pylint: enable=protected-access
