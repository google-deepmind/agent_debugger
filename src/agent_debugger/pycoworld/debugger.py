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

"""Specific implementation of the debugger for pycoworld."""

from agent_debugger.src.agent_debugger import agent as debugger_agent
from agent_debugger.src.agent_debugger import debugger as dbg
from agent_debugger.src.agent_debugger.pycoworld import extractors as pycolab_extractors
from agent_debugger.src.agent_debugger.pycoworld import interventions as pycolab_interventions
from agent_debugger.src.pycoworld import serializable_environment


class PycoworldDebugger(dbg.Debugger):
  """Overriding the Debugger with additional interventions and extractors.

  Attributes:
    interventions: An object to intervene on pycoworld nodes.
    extractors: An object to extract information from pycoworld nodes.
  """

  def __init__(
      self,
      agent: debugger_agent.DebuggerAgent,
      env: serializable_environment.SerializableEnvironment,
  ) -> None:
    """Initializes the object."""
    super().__init__(agent, env)

    self.interventions = pycolab_interventions.PycoworldInterventions(self._env)
    self.extractors = pycolab_extractors.PycoworldExtractors()
