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

"""Custom types used throughout the codebase.

A seed is required for the agent. For the environment, it can be included
but it's not mandatory in the dm_env standard.
"""

from typing import Any, Callable, NamedTuple

import dm_env

from agent_debugger.src.pycoworld import serializable_environment

Action = Any
Observation = Any
TimeStep = dm_env.TimeStep
FirstTimeStep = dm_env.StepType.FIRST
MidTimeStep = dm_env.StepType.MID
LastTimeStep = dm_env.StepType.LAST
Environment = dm_env.Environment
SerializableEnvironment = serializable_environment.SerializableEnvironment
Agent = Any
Rng = Any
EnvState = Any
EnvBuilder = Callable[[], Environment]
SerializableEnvBuilder = Callable[[], SerializableEnvironment]


# We don't use dataclasses as they are not supported by jax.
class AgentState(NamedTuple):
  internal_state: Any
  seed: Rng
