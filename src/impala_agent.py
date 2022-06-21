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

"""Impala agent class, which follows the DebuggerAgent interface."""

from typing import Tuple, Any, Callable

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

from agent_debugger.src.agent_debugger import agent
from agent_debugger.src.agent_debugger import types


class ImpalaAgent(agent.DebuggerAgent):
  """Impala agent class."""

  def __init__(
      self,
      net_factory: Callable[[], hk.RNNCore],
      params: hk.Params,
  ) -> None:
    """Initializes the agent.

    Args:
      net_factory: Function to create the model.
      params: The parameters of the agent.
    """
    _, self._initial_state = hk.transform(
        lambda batch_size: net_factory().initial_state(batch_size))

    self._init_fn, apply_fn = hk.without_apply_rng(
        hk.transform(lambda obs, state: net_factory().__call__(obs, state)))
    self._apply_fn = jax.jit(apply_fn)

    self._params = params

  def initial_state(self, rng: jnp.ndarray) -> types.AgentState:
    """Returns the agent initial state."""
    # Wrapper method to avoid pytype attribute-error.
    return types.AgentState(
        internal_state=self._initial_state(self._params, rng, batch_size=1),
        seed=rng)

  def step(
      self,
      timestep: dm_env.TimeStep,
      state: types.AgentState,
  ) -> Tuple[types.Action, Any, types.AgentState]:
    """Steps the agent in the environment."""
    net_out, next_state = self._apply_fn(self._params, timestep,
                                         state.internal_state)

    # Sample an action and return.
    action = hk.multinomial(state.seed, net_out.policy_logits, num_samples=1)
    action = jnp.squeeze(action, axis=-1)
    action = int(action)

    new_rng, _ = jax.random.split(state.seed)
    new_agent_state = types.AgentState(internal_state=next_state, seed=new_rng)
    return action, net_out, new_agent_state
