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

"""Pycoworld environment.

Pycolab only provides a game engine, i.e. a way to create a game, with the
different objects and their dynamics. However, we'd like to use a more formal
interface which is an environment, with a reset(), step(), action_spec() and
observation_spec() methods. We need to create an object to wrap the engine
and create a true environment, inheriting dm_env.Environment.

This file also contains a build_environment method, which takes a pycoworld
level name, and directly creates the associated environment, ready to be
used in association with the agent.
"""

import copy
import dataclasses
import itertools
from typing import Any, Callable, Optional, Union

import dm_env
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from pycolab import cropping as pycolab_cropping
from pycolab import engine as pycolab_engine
from pycolab import rendering
import tree

from agent_debugger.src.pycoworld import default_constants
from agent_debugger.src.pycoworld import serializable_environment
from agent_debugger.src.pycoworld.levels import base_level
from agent_debugger.src.pycoworld.levels import level_names

_TILES = list(range(len(default_constants.Tile)))
_PLAYER_TILE = chr(8)

_RGB_MAPPING = {chr(n): (4 * n, 4 * n, 4 * n) for n in _TILES}


class ObservationDistiller:
  """This class modifies the observations before they are sent to the agent."""

  def __init__(
      self,
      array_converter: rendering.ObservationToArray,
      cropping: Optional[pycolab_cropping.ObservationCropper] = None
  ) -> None:
    """Initializes a Distiller."""
    self._array_converter = array_converter
    cropper = pycolab_cropping.ObservationCropper
    self._cropping = tree.map_structure(
        lambda c: cropper() if c is None else c, cropping)

  def set_engine(self, engine: pycolab_engine.Engine) -> None:
    """Informs the Distiller of the current game engine in use.

    Args:
      engine: current engine in use by the `Environment` adapter.
    """
    tree.map_structure(lambda c: c.set_engine(engine), self._cropping)

  def __call__(self, observation: Any) -> Any:
    """Distills a pycolab observation into the format required by the agent.

    Args:
      observation: observation to distill.

    Returns:
      the observation distilled for supplying to the agent.
    """
    return tree.map_structure(
        lambda c: self._array_converter(c.crop(observation)), self._cropping)


@dataclasses.dataclass
class EnvState:
  """The Pycoworld environment state."""
  state: Any
  current_game: pycolab_engine.Engine
  game_over: bool
  observation_distiller: ObservationDistiller
  seed: int


class PycoworldEnvironment(serializable_environment.SerializableEnvironment):
  """A wrapper around the for_python.Environment to get and set states.

  The serialization is just a copy of the wrapped environment.
  """

  def __init__(
      self,
      seed: int,
      game_factory: Callable[[jnp.ndarray], pycolab_engine.Engine],
      possible_actions: set[int],
      default_reward: Optional[float],
      observation_distiller: ObservationDistiller,
      max_iterations: Optional[int] = None,
      max_iterations_discount: Optional[float] = None,
  ) -> None:
    """Initializes the pycoworld environment."""
    self._game_factory = game_factory
    self._default_reward = default_reward
    self._observation_distiller = observation_distiller
    self._max_iterations = max_iterations
    self._max_iterations_discount = max_iterations_discount

    self._state = None
    self._current_game = None
    self._game_over = None
    self._seed = seed

    # Compute action and observation spec.
    self._action_spec = dm_env.specs.DiscreteArray(
        len(possible_actions), dtype='int32', name='discrete')
    self._observation_spec = self._compute_observation_spec()

  def reset(self) -> dm_env.TimeStep:
    """Starts a new episode."""
    self._current_game = self._game_factory(jrandom.PRNGKey(self._seed))
    self._state = dm_env.StepType.FIRST
    self._observation_distiller.set_engine(self._current_game)
    # Collect environment returns from starting the game and update state.
    observations, reward, discount = self._current_game.its_showtime()

    self._game_over = self._is_game_over()
    observations, _, _ = self._standardise_timestep_values(
        observations, reward, discount)
    return dm_env.TimeStep(
        step_type=self._state,
        reward=None,
        discount=None,
        observation=observations)

  def step(self, action: int) -> dm_env.TimeStep:
    """Applies action, steps the world forward, and returns observations."""
    # Clear episode internals and start a new episode, if episode ended or if
    # the game was not already underway.
    if self._state == dm_env.StepType.LAST:
      self._drop_last_episode()
    if self._current_game is None:
      return self.reset()

    # Execute the action in pycolab.
    observations, reward, discount = self._current_game.play(action)

    self._game_over = self._is_game_over()
    observations, reward, discount = self._standardise_timestep_values(
        observations, reward, discount)

    # Check the current status of the game.
    if self._game_over:
      self._state = dm_env.StepType.LAST
    else:
      self._state = dm_env.StepType.MID

    return dm_env.TimeStep(
        step_type=self._state,
        reward=reward,
        discount=discount,
        observation=observations)

  def observation_spec(self) -> dm_env.specs.Array:
    """Returns the observation specifications of the environment."""
    return self._observation_spec

  def action_spec(self) -> dm_env.specs.Array:
    """Returns the action specifications of the environment."""
    return self._action_spec

  def get_state(self) -> EnvState:
    """Returns the state of the pycoworld environment."""
    env_state = EnvState(
        state=self._state,
        current_game=self._current_game,
        game_over=self._game_over,
        observation_distiller=self._observation_distiller,
        seed=self._seed)
    return copy.deepcopy(env_state)

  def set_state(self, env_state: EnvState) -> None:
    """Sets the state of the pycoworld environment."""
    env_state = copy.deepcopy(env_state)
    self._state = env_state.state
    self._current_game = env_state.current_game
    self._game_over = env_state.game_over
    self._observation_distiller = env_state.observation_distiller
    self._seed = env_state.seed

  def _compute_observation_spec(self) -> Any:
    """Returns after compute the observation spec of the environment.

    We need a special method to do this, as we are retrieving the specs by
    launching a game. This is an internal and private method only.
    """

    def obs_names_that_count_up():
      for i in itertools.count():
        yield 'obs{}'.format(i)

    names = obs_names_that_count_up()

    timestep = self.reset()
    spec_array = dm_env.specs.Array
    observation_spec = tree.map_structure(
        lambda a: spec_array(shape=a.shape, dtype=a.dtype, name=next(names)),
        timestep.observation)
    self._drop_last_episode()
    return observation_spec

  def _standardise_timestep_values(
      self,
      observations: Any,
      reward: Optional[float],
      discount: Optional[float],
  ) -> tuple[Any, Optional[float], Optional[float]]:
    """Applies defaults and standard packaging to timestep values if needed."""
    observations = copy.deepcopy(self._observation_distiller(observations))
    if isinstance(observations, np.ndarray):
      observations = {'board': observations}
    reward = reward if reward is not None else self._default_reward
    if self._max_iterations is not None and (
        self._current_game.the_plot.frame >=
        self._max_iterations) and (not self._current_game.game_over):
      if self._max_iterations_discount is not None:
        discount = self._max_iterations_discount
    return observations, reward, discount

  def _is_game_over(self) -> bool:
    """Returns whether the game is over."""
    # If we've reached the maximum number of game iterations, terminate the
    # current game.
    if self._max_iterations is not None and (self._current_game.the_plot.frame
                                             >= self._max_iterations):
      return True
    return self._current_game.game_over

  def _drop_last_episode(self) -> None:
    """Clears all the internal information about the game."""
    self._state = None
    self._current_game = None
    self._game_over = None


def build_environment(
    level: Union[base_level.PycoworldLevel, str],
    seed: int = 1,
    episode_length: Optional[int] = None,
    observation_encoding: str = 'feature_map',
    egocentric_horizon: Optional[int] = None,
) -> PycoworldEnvironment:
  """Builds and returns the environment for pycoworld.

  The observation is either a feature map (one hot encoding), a raw (the tile id
  directly) or rgb map of the board of the game. The actions are discrete and
  you can get the bounds via the env specs.
  Args:
    level: The level to build.
    seed: The seed used by the level
    episode_length: Number of steps in the episode
    observation_encoding: Must be in {feature_map, board, rgb}
    egocentric_horizon: The sight distance of the agent. None means the agent
      sees the complete board.

  Returns:
    The environment for the game
  """
  if isinstance(level, str):
    level = level_names.level_by_name(level)

  # The distiller is used to processed the raw observation provided by pycolab
  # to an observation fed to the agent, here a numpy binary array.
  if observation_encoding == 'feature_map':
    array_converter = rendering.ObservationToFeatureArray(
        layers=list(map(chr, _TILES)), permute=(1, 2, 0))
  elif observation_encoding == 'board':
    no_mapping = {chr(x): x for x in _TILES}
    array_converter = rendering.ObservationToArray(no_mapping, dtype=np.uint8)
  elif observation_encoding == 'rgb':
    array_converter = rendering.ObservationToArray(
        _RGB_MAPPING, permute=(1, 2, 0), dtype=np.uint8)
  else:
    raise ValueError('Observation encoding must be in {feature_map, rgb},'
                     f'got {observation_encoding}.')

  if egocentric_horizon is not None:
    square_side_length = 2 * egocentric_horizon + 1
    cropper = pycolab_cropping.ScrollingCropper(
        rows=square_side_length,
        cols=square_side_length,
        to_track=[_PLAYER_TILE],
        pad_char=chr(0),
        scroll_margins=(1, 1))
  else:
    cropper = None

  observation_distiller = ObservationDistiller(
      array_converter=array_converter, cropping=cropper)

  return PycoworldEnvironment(
      seed=seed,
      game_factory=level.sample_game,
      possible_actions=set(level.actions),
      observation_distiller=observation_distiller,
      default_reward=0.,
      max_iterations=episode_length,
      max_iterations_discount=1.)
