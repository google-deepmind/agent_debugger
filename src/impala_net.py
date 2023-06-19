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

"""Neural network used in Impala trained agents."""

from typing import NamedTuple, Optional, Sequence, Union

import dm_env
import haiku as hk
import jax.nn
import jax.numpy as jnp

NetState = Union[jnp.ndarray, hk.LSTMState]


# Kept as a NamedTuple as jax does not support dataclasses.
class NetOutput(NamedTuple):
  """Dataclass to define network outputs."""
  policy_logits: jnp.ndarray
  value: jnp.ndarray


class RecurrentConvNet(hk.RNNCore):
  """A class for Impala nets.

  Architecture:
  MLP torso -> LSTM -> MLP head -> (linear policy logits, linear value)

  If initialised with lstm_width=0, skips the LSTM layer.
  """

  def __init__(
      self,
      num_actions: int,
      conv_widths: Sequence[int],
      conv_kernels: Sequence[int],
      padding: str,
      torso_widths: Sequence[int],
      lstm_width: Optional[int],
      head_widths: Sequence[int],
      name: str = None,
  ) -> None:
    """Initializes the impala net."""
    super().__init__(name=name)
    self._num_actions = num_actions
    self._torso_widths = torso_widths
    self._head_widths = head_widths
    self._core = hk.LSTM(lstm_width) if lstm_width else None

    conv_layers = []
    for width, kernel_size in zip(conv_widths, conv_kernels):
      layer = hk.Conv2D(
          width, kernel_shape=[kernel_size, kernel_size], padding=padding)
      conv_layers += [layer, jax.nn.relu]
    self._conv_net = hk.Sequential(conv_layers + [hk.Flatten()])

  def initial_state(self, batch_size: int) -> NetState:
    """Returns a fresh hidden state for the LSTM core."""
    return self._core.initial_state(batch_size) if self._core else jnp.zeros(())

  def __call__(
      self,
      x: dm_env.TimeStep,
      state: NetState,
  ) -> tuple[NetOutput, NetState]:
    """Steps the net, applying a forward pass of the neural network."""
    # Apply torso.
    observation = x.observation['board'].astype(dtype=jnp.float32) / 255
    observation = jnp.expand_dims(observation, axis=0)
    output = self._torso(observation)

    if self._core is not None:
      output, state = self._core(output, state)

    policy_logits, value = self._head(output)
    return NetOutput(policy_logits=policy_logits[0], value=value[0]), state

  def _head(self, activations: jnp.ndarray) -> jnp.ndarray:
    """Returns new activations after applying the head network."""
    pre_outputs = hk.nets.MLP(self._head_widths)(activations)
    policy_logits = hk.Linear(self._num_actions)(pre_outputs)
    value = hk.Linear(1)(pre_outputs)
    value = jnp.squeeze(value, axis=-1)
    return policy_logits, value

  def _torso(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Returns activations after applying the torso to the inputs."""
    return hk.Sequential([self._conv_net,
                          hk.nets.MLP(self._torso_widths)])(
                              inputs)
