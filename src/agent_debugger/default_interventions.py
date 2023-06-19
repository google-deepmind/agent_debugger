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

"""Base interventions for the debugger."""

import copy
import dataclasses
from typing import Union

import jax

from agent_debugger.src.agent_debugger import node as node_lib
from agent_debugger.src.agent_debugger import types
from agent_debugger.src.pycoworld import serializable_environment


class DefaultInterventions:
  """Object containing the default debugger interventions."""

  def __init__(
      self,
      env: serializable_environment.SerializableEnvironment,
  ) -> None:
    """Initializes the default interventions object.

    Args:
      env: The environment to be able to reset after a seed intervention.
    """
    self._env = env

  def change_agent_seed(
      self,
      node: node_lib.Node,
      seed: Union[int, types.Rng],
  ) -> node_lib.Node:
    """Intervenes to replace the agent seed, used for acting.

    Args:
     node: The node to intervene on.
     seed: The new seed to put to the agent state of the node. Can be int or
       jax.random.PRNGKey directly.

    Returns:
     A new node, whose agent seed has been updated.
    """
    if isinstance(seed, int):
      seed = jax.random.PRNGKey(seed)
    new_agent_state = node.agent_state._replace(seed=seed)
    return dataclasses.replace(node, agent_state=new_agent_state)

  def change_env_seed(
      self,
      node: node_lib.Node,
      seed: int,
  ) -> node_lib.Node:
    """Intervenes to replace the seed of the environment.

    The state of the environment must contain an attribute 'seed' to be able to
    use this intervention.

    Args:
      node: The node to intervene on.
      seed: The new seed to put to the environment state of the node.

    Returns:
      A new node, which environment state has been modified.
    """
    # Create the new state.
    state = copy.deepcopy(node.env_state)
    state.seed = seed
    self._env.set_state(state)

    new_timestep = self._env.reset()
    new_state = self._env.get_state()
    return dataclasses.replace(
        node, env_state=new_state, last_timestep=new_timestep)

  def change_agent_next_actions(
      self,
      node: node_lib.Node,
      forced_next_actions: list[types.Action],
  ) -> node_lib.Node:
    """Changes the next actions of the agent at a given node.

    This intervention allows the user to change the N next actions of the agent,
    not only the next one. When stepping the debugger from this node, the list
    for the next node is actualised by removing the first action (since it has
    just been executed).

    Args:
      node: The node to intervene on.
      forced_next_actions: The next actions to be taken by the agent.

    Returns:
      A new node, which, when stepped from, will force the N next actions taken
      by the agent.
    """
    return dataclasses.replace(node, forced_next_actions=forced_next_actions)
