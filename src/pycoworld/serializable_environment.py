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

"""Serializable environment class.

The main feature is to ability to retrieve and set the state of an environment.
"""

import abc
from typing import Any

import dm_env


class SerializableEnvironment(dm_env.Environment, abc.ABC):
  """Abstract class with methods to set and get states.

  The state can be anything even though we prefer mappings like dicts for
  readability. It must contain all information, no compression is allowed. The
  state must be a parse of the environment containing only the stateful
  variables.
  There is no assumption on how the agent would use this object: please provide
  copies of internal attributes to avoid side effects.
  """

  @abc.abstractmethod
  def get_state(self) -> Any:
    """Returns the state of the environment."""

  @abc.abstractmethod
  def set_state(self, state: Any) -> Any:
    """Sets the state of the environment.

    Args:
      state: The state to set.
    """
