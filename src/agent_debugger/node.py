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

"""The base node interface used in the debugger."""

import dataclasses
from typing import Any, Optional, Sequence

from agent_debugger.src.agent_debugger import types


@dataclasses.dataclass
class Node:
  """A node is the concatenation of the agent_state and the env_state.

  Besides the agent and env state, it also carries the action taken by the agent
  and the timestep returned by the environment in the last transition. This
  transition can be seen as the arrow leading to the state, therefore we call it
  'last_action' and 'last_timestep'. The latter will be used to step to the next
  state.
  Finally, the node also carries some future agent actions that the user wants
  to enforce. The first element of the list will be used to force the next
  transition action, and the list minus this first element will be passed to the
  next state.
  """

  agent_state: types.AgentState
  last_action: types.Action
  env_state: types.EnvState
  last_timestep: types.TimeStep
  forced_next_actions: Sequence[types.Action] = ()
  episode_step: int = 0
  last_agent_output: Optional[Any] = None

  @property
  def is_terminal(self) -> bool:
    """Returns whether the last timestep is of step_type LAST."""
    return self.last_timestep.step_type == types.LastTimeStep
