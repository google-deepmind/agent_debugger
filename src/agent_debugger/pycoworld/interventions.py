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

"""Pycoworld interventions.

These are methods of the form (node, **params) -> new_node.
"""

import copy
import dataclasses
from typing import Any, Type

import dm_env

from agent_debugger.src.agent_debugger import default_interventions
from agent_debugger.src.agent_debugger import node as node_lib
from agent_debugger.src.agent_debugger.pycoworld import types


def _check_positive_position(position: types.Position) -> None:
  """Checks if the position is positive, required by pycolab."""
  if position.row < 0 or position.col < 0:  # pytype: disable=attribute-error  # enable-nested-classes
    raise ValueError('The new position must have positive coordinates. Got '
                     f'{position}.')


class InterventionContext:
  """Context used for interventions, exposing the engine and postprocessing.

  Attributes:
    engine: The internal Pycolab game engine, useful to perform interventions.
    new_node: The node after intervention, which will be returned.
  """

  def __init__(self, node: node_lib.Node):
    """Initializes."""
    self._node = node
    self._env_state = copy.deepcopy(self._node.env_state)
    self.engine = self._env_state.current_game
    self.new_node = None

  def __enter__(self) -> Type['InterventionContext']:
    """Returns the object itself when we enter the context."""
    return self

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    """Applies some postprocessing and computes the new node to return."""
    self._env_state.current_game._render()

    # Compute the new timestep that the agent will see.
    new_observation = copy.deepcopy(self._node.last_timestep.observation)
    new_observation['board'] = self._env_state.observation_distiller(
        self._env_state.current_game._board)
    new_timestep = dm_env.TimeStep(
        step_type=self._node.last_timestep.step_type,
        reward=self._node.last_timestep.reward,
        discount=self._node.last_timestep.discount,
        observation=new_observation)

    self.new_node = dataclasses.replace(
        self._node, env_state=self._env_state, last_timestep=new_timestep)


class PycoworldInterventions(default_interventions.DefaultInterventions):
  """Object containing the pycolab interventions."""

  # pylint: disable=protected-access
  def set_sprite_visibility(
      self,
      node: node_lib.Node,
      sprite_id: int,
      visibility: bool,
  ) -> node_lib.Node:
    """Intervenes on a pycolab environment state to make a sprite (in)visible.

    Args:
      node: The node to intervene on.
      sprite_id: The identifier of the sprite. A full list can be found in the
        pycoworld/default_constants.py file.
      visibility: Whether the sprite should be made visible or invisible.

    Returns:
      A new node, where the sprite is (in)visible.
    """
    with InterventionContext(node) as context:
      sprite = context.engine._sprites_and_drapes[chr(sprite_id)]
      sprite._visible = visibility
    return context.new_node

  def move_sprite_to(
      self,
      node: node_lib.Node,
      sprite_id: int,
      dest_position: types.Position,
  ) -> node_lib.Node:
    """Intervenes on a pycolab environment state to move a sprite.

    Args:
      node: The node to intervene on.
      sprite_id: The identifier of the sprite. A full list can be found in the
        pycoworld/default_constants.py file.
      dest_position: The desired position of the sprite.

    Returns:
      A new node, where the sprite has been moved.
    """
    with InterventionContext(node) as context:
      dest_position = types.to_pycolab_position(dest_position)
      _check_positive_position(dest_position)
      sprite = context.engine._sprites_and_drapes[chr(sprite_id)]
      sprite._position = dest_position
    return context.new_node

  def move_drape_element_to(
      self,
      node: node_lib.Node,
      drape_id: int,
      start_position: types.Position,
      dest_position: types.Position,
  ) -> node_lib.Node:
    """Intervenes on a pycolab environment state to move a drape.

    Args:
      node: The node to intervene on.
      drape_id: The identifier of the drape. A full list can be found in the
        pycoworld/default_constants.py file.
      start_position: The position of the element to replace, coordinates must
        be positive.
      dest_position: The destination position of the element, coordinates must
        be positive.

    Returns:
      A new node, which drape has been updated.
    """
    with InterventionContext(node) as context:
      dest_position = types.to_pycolab_position(dest_position)
      start_position = types.to_pycolab_position(start_position)
      _check_positive_position(start_position)
      _check_positive_position(dest_position)
      drape = context.engine._sprites_and_drapes[chr(drape_id)]
      drape._c_u_r_t_a_i_n[start_position] = False
      drape._c_u_r_t_a_i_n[tuple(dest_position)] = True
    return context.new_node

  def remove_drape_element(
      self,
      node: node_lib.Node,
      drape_id: int,
      position: types.Position,
  ) -> node_lib.Node:
    """Intervenes to remove an element in a drape.

    Args:
      node: The node to intervene on.
      drape_id: The identifier of the drape. A full list can be found in the
        pycoworld/default_constants.py file.
      position: The position of the element to remove, coordinates must be
        positive.

    Returns:
      A new node, which drape has been updated.
    """
    with InterventionContext(node) as context:
      position = types.to_pycolab_position(position)
      _check_positive_position(position)
      drape = context.engine._sprites_and_drapes[chr(drape_id)]
      drape._c_u_r_t_a_i_n[position] = False
    return context.new_node

  def add_drape_element(
      self,
      node: node_lib.Node,
      drape_id: int,
      position: types.Position,
  ) -> node_lib.Node:
    """Intervenes to add an element in a drape.

    Args:
      node: The node to intervene on.
      drape_id: The identifier of the drape. A full list can be found in the
        pycoworld/default_constants.py file.
      position: The position of the element to add, coordinates must be
        positive.

    Returns:
      A new node, which drape has been updated.
    """
    with InterventionContext(node) as context:
      position = types.to_pycolab_position(position)
      _check_positive_position(position)
      drape = context.engine._sprites_and_drapes[chr(drape_id)]
      drape._c_u_r_t_a_i_n[position] = True
    return context.new_node

  def replace_backdrop_element(
      self,
      node: node_lib.Node,
      position: types.Position,
      new_element_id: int,
  ) -> node_lib.Node:
    """Intervenes to replace an element of the backdrop.

    Args:
      node: The node to intervene on.
      position: The position of the element to replace, coordinates must be
        positive.
      new_element_id: The new element id to put at this position.

    Returns:
      A new node, where the backdrop has been updated.
    """
    with InterventionContext(node) as context:
      position = types.to_pycolab_position(position)
      _check_positive_position(position)
      context.engine._backdrop._p_a_l_e_t_t_e._legal_characters.add(
          chr(new_element_id))
      context.engine._backdrop.curtain[tuple(position)] = new_element_id
    return context.new_node

  def set_frame_count(
      self,
      node: node_lib.Node,
      frame_count: int,
  ) -> node_lib.Node:
    """Changes the internal frame number of a pycolab engine.

    This number influences the 'plot' object, which contains the frame number as
    an attribute.
    Args:
      node: The node to intervene on.
      frame_count: the frame countto set on the engine. It must be positive.

    Returns:
      A new node, where the backdrop has been updated.
    """
    with InterventionContext(node) as context:
      if frame_count < 0:
        raise ValueError(f'The frame count must be positive. Got {frame_count}')
      context.engine._the_plot._frame = frame_count
    return context.new_node

  # pylint: enable=protected-access
