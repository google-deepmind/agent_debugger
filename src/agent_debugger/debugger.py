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

"""Debugger object, main interface of the framework."""

import contextlib
import operator
from typing import Any, Callable, List, Optional, Sequence, Tuple

import jax.random as jrandom

from agent_debugger.src.agent_debugger import agent as debugger_agent
from agent_debugger.src.agent_debugger import default_interventions
from agent_debugger.src.agent_debugger import node as node_lib
from agent_debugger.src.pycoworld import serializable_environment


class NotReachedError(RuntimeError):
  """Raised when a rollout did not reach the required point."""


class Debugger:
  """Debugger to analyze agents in an environment step by step.

  The debugger can be used to step an agent in an environment, intervene on the
  agent and environment, and extract information. The debugger object is the
  main API of the Agent Debugger.

  Attributes:
    interventions: The interventions module to intervene on nodes.
  """

  def __init__(
      self,
      agent: debugger_agent.DebuggerAgent,
      env: serializable_environment.SerializableEnvironment,
  ) -> None:
    """Initializes the debugger with an agent and an environment."""
    self._agent = agent
    self._env = env
    self._rollout_mode = False
    self._is_first_rollout_step = False

    self.interventions = default_interventions.DefaultInterventions(self._env)

  def get_root_node(self, seed: int = 1) -> node_lib.Node:
    """Returns a root node, containing an initial world state.

    Args:
      seed: The seed to use to sample the world state.
    """
    rng = jrandom.PRNGKey(seed=seed)
    timestep = self._env.reset()
    env_state = self._env.get_state()
    agent_state = self._agent.initial_state(rng)
    return node_lib.Node(
        agent_state=agent_state,
        env_state=env_state,
        last_action=None,
        last_timestep=timestep)

  @contextlib.contextmanager
  def rollout_mode(self):
    """Context manager for the rollout mode.

    Being in rollout mode means some performance optimisation will be done when
    we step the agent and the environment together. Specifically, to avoid
    getting and setting their states at each step of the rollout rather than
    just once at the beginning, we pass an option to deactivate it if the
    rollout mode is on.
    In practice, this mode reduces the compute time of debugger.get_rollout
    while returning exactly the same result.

    Yields:
      None. The method is used inside the same object, so no need to yield any
      reference.
    """
    self._rollout_mode = True
    self._is_first_rollout_step = True
    try:
      yield None
    finally:
      self._rollout_mode = False
      self._is_first_rollout_step = False

  def step_forward(self, node: node_lib.Node) -> node_lib.Node:
    """Steps from a world state, using the environment and agent dynamics.

    Args:
      node: The node to step from.

    Returns:
      The new node obtained after stepping.
    """
    action, agent_output, new_agent_state = self._agent.step(
        node.last_timestep, node.agent_state)

    if node.forced_next_actions:
      action = node.forced_next_actions[0]

    if not self._rollout_mode or self._is_first_rollout_step:
      self._env.set_state(node.env_state)
    timestep = self._env.step(action)
    new_env_state = self._env.get_state()
    self._is_first_rollout_step = False

    return node_lib.Node(
        agent_state=new_agent_state,
        env_state=new_env_state,
        last_action=action,
        last_timestep=timestep,
        forced_next_actions=node.forced_next_actions[1:],
        last_agent_output=agent_output,
        episode_step=node.episode_step + 1)

  def get_rollout(
      self,
      initial_node: node_lib.Node,
      maximum_length: int,
      endpoint: Optional[Callable[[node_lib.Node],
                                  bool]] = operator.attrgetter('is_terminal'),
      raise_endpoint_not_reached: bool = False,
  ) -> List[node_lib.Node]:
    """Returns the list of nodes obtained by stepping from an initial node.

    Args:
      initial_node: The initial node to step from. It is the first element of
        the returned list.
      maximum_length: The maximum length of the returned list.
      endpoint: Function that specifies when to stop the rollout. Defaulted to
        is_terminal. If None, the rollout stops after maximum_length iterations,
        and some intermediate nodes in the list can therefore be terminal. If we
        fell upon a terminal node in this case, we just keep on stepping the
        environment, which must return a FIRST node, following the dm_env API.
      raise_endpoint_not_reached: Whether to raise an error if the endpoint is
        not reached within the maximum rollout length.

    Returns:
      A list of nodes.

    Raises:
      NotReachedError: If raise_endpoint_not_reached = True and the
        endpoint was not reached within the maximum rollout length.
      ValueError: If raise_endpoint_not_reached=True while endpoint=None.
    """
    if endpoint is None and raise_endpoint_not_reached:
      raise ValueError(
          'Cannot raise endpoint not reached error when endpoint is None.')
    endpoint_reached = False
    with self.rollout_mode():
      visited_nodes = [initial_node]
      for _ in range(maximum_length - 1):
        current_node = self.step_forward(visited_nodes[-1])
        visited_nodes.append(current_node)
        if endpoint is not None and endpoint(current_node):
          endpoint_reached = True
          break

    if raise_endpoint_not_reached and not endpoint_reached:
      raise NotReachedError(
          'Rollout did not reach the end point within the maximum length.')

    return visited_nodes

  def get_intervened_rollout(
      self,
      initial_node: node_lib.Node,
      maximum_length: int,
      rollout_breakpoint: Callable[[node_lib.Node], bool],
      intervention_at_breakpoint: Callable[[node_lib.Node], node_lib.Node],
      endpoint: Optional[Callable[[node_lib.Node],
                                  bool]] = operator.attrgetter('is_terminal'),
      raise_breakpoint_not_reached: bool = True,
  ) -> List[node_lib.Node]:
    """Returns a modified version of get_rollout, with an intervention.

    The user can provide a breakpoint function to branch at a given point in
    the rollout. For now it is possible to branch only once in the rollout.
    This is a separate function from get_rollout because we must set and unset
    the rollout_mode of the PTS during the intervention.
    Args:
      initial_node: The initial node to step from. It is the first element of
        the returned list.
      maximum_length: The maximum length of the returned list.
      rollout_breakpoint: Function that specifies when to branch in the rollout,
        to intervene. If None, we never branch.
      intervention_at_breakpoint: The intervention to perform when the
        breakpoint is reached.
      endpoint: See get_rollout.
      raise_breakpoint_not_reached: Whether to raise an error if the breakpoint
        is not reached within the maximum rollout length.

    Returns:
      A list of nodes, where an intervention has been performed on one of them.

    Raises:
      NotReachedError: If raise_breakpoint_not_reached = True and the
        breakpoint was not reached within the maximum rollout length.
      ValueError: If raise_breakpoint_not_reached=True while endpoint=None.
    """
    visited_nodes = self.get_rollout(
        initial_node,
        maximum_length,
        endpoint=rollout_breakpoint,
        raise_endpoint_not_reached=raise_breakpoint_not_reached)
    visited_nodes[-1] = intervention_at_breakpoint(visited_nodes[-1])
    visited_nodes.extend(
        self.get_rollout(
            initial_node=visited_nodes[-1],
            maximum_length=maximum_length,
            endpoint=endpoint)[1:])
    return visited_nodes

  def get_actions_rewards(
      self,
      rollout: Sequence[node_lib.Node],
      cast_action: Callable[[Any], Any] = lambda x: x
  ) -> Tuple[Sequence[Optional[Any]], Sequence[float]]:
    """Returns the actions and rewards of a rollout in a nice format.

    Args:
      rollout: The rollout of nodes to be parsed.
      cast_action: A function to cast the actions to the right format.
    """
    curate = lambda x: cast_action(x) if x is not None else x
    actions = [curate(node.last_action) for node in rollout]
    rewards = [node.last_timestep.reward for node in rollout]
    return actions, rewards
