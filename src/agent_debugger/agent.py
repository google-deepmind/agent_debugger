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

"""Base class of agent recognized by the Agent Debugger."""

import abc
from typing import Any, Tuple

import chex
import dm_env

from agent_debugger.src.agent_debugger import types


class DebuggerAgent(abc.ABC):
  """The standard agent interface to be used in the debugger.

  The internal state is wrapped with the seed. The step method takes an
  observation and not a timestep. The agent parameters (mostly neural network
  parameters) are fixed and are NOT part of the state.
  """

  @abc.abstractmethod
  def initial_state(self, rng: chex.PRNGKey) -> types.AgentState:
    """Returns the initial state of the agent."""

  @abc.abstractmethod
  def step(
      self,
      timestep: dm_env.TimeStep,
      state: types.AgentState,
  ) -> Tuple[types.Action, Any, types.AgentState]:
    """Steps the agent, taking a timestep (including observation) and a state."""
