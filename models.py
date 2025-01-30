from typing import Dict, Optional, Tuple
import jax
import numpy as np
from jax import numpy as jnp
from jax import random, jit
import flax.linen as nn
from flax.linen.module import compact, nowrap
from flax.typing import Array, PRNGKey, Callable
from dataclasses import field
import utils


class AudioRNN(nn.Module):
    hidden_size: int
    cell_type: type(nn.RNNCellBase)
    cell_args: Optional[Dict] = field(default_factory=dict)
    residual_connection: bool = True
    out_channels: int = 1
    dtype: type = jnp.float32

    def setup(self):
        self.rec = nn.RNN(self.cell_type(self.hidden_size, dtype=self.dtype, param_dtype=self.dtype, **self.cell_args))
        self.linear = nn.Dense(self.out_channels)

    @nn.compact
    def __call__(self, carry, x):
        new_carry, states = self.rec(x, initial_carry=carry, return_carry=True)
        out = self.linear(states)
        if self.residual_connection:
            out += x[..., 0:1]
        return new_carry, out

    def initialise_carry(self, input_shape):
        return self.cell_type(self.hidden_size, parent=None, dtype=self.dtype, param_dtype=self.dtype, **self.cell_args).initialize_carry(jax.random.key(0), input_shape)


class FIRInterpLSTMCell(nn.LSTMCell):
    kernel: Array = None
    skip: int = 0

    @compact
    def __call__(self, carry, inputs):
        c, h = carry
        c_interp = jnp.matmul(c[..., self.skip:], self.kernel)
        h_interp = jnp.matmul(h[..., self.skip:], self.kernel)
        (latest_c, latest_h), _ = super().__call__((c_interp, h_interp), inputs)
        new_c = c.at[..., -1].set(latest_c)
        new_h = h.at[..., -1].set(latest_h)
        new_c = jnp.roll(new_c, shift=1, axis=-1)
        new_h = jnp.roll(new_h, shift=1, axis=-1)
        return (new_c, new_h), latest_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        num_coeffs = jnp.size(self.kernel)
        mem_shape = batch_dims + (self.features, num_coeffs + self.skip)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)

class DelayLineLSTMCell(nn.LSTMCell):
    delay: int = 2

    @compact
    def __call__(self, carry, inputs):
        c, h = carry
        (latest_c, latest_h), _ = super().__call__((c[..., -1], h[..., -1]), inputs)
        new_c = c.at[..., -1].set(latest_c)
        new_h = h.at[..., -1].set(latest_h)
        new_c = jnp.roll(new_c, shift=1, axis=-1)
        new_h = jnp.roll(new_h, shift=1, axis=-1)
        return (new_c, new_h), latest_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        mem_shape = batch_dims + (self.features, self.delay)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)


class FIRInterpRNNCell(nn.SimpleCell):
    kernel: Array = None
    skip: int = 0

    @compact
    def __call__(self, carry, inputs):
        h = carry
        h_interp = jnp.matmul(h[..., self.skip:], self.kernel)
        latest_h, _ = super().__call__(h_interp, inputs)
        new_h = h.at[..., -1].set(latest_h)
        new_h = jnp.roll(new_h, shift=1, axis=-1)
        return new_h, latest_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        num_coeffs = jnp.size(self.kernel)
        mem_shape = batch_dims + (self.features, num_coeffs + self.skip)
        h = self.carry_init(key1, mem_shape, self.param_dtype)
        return h


def audio_rnn_inference(model_info: Dict,
                       params: Dict,
                       in_sig: jax.Array) -> jax.Array:
    if in_sig.ndim == 2:
        in_sig = jnp.expand_dims(in_sig, 0)
    elif in_sig.ndim == 1:
        in_sig = jnp.expand_dims(in_sig, (0, 2))

    if model_info['unit_type'] == 'RNN':
        model = AudioRNN(cell_type=FIRInterpRNNCell,
                         hidden_size=model_info['hidden_size'],
                         cell_args={'kernel': np.array([1.0])})
    else:
        delay = model_info['delay']
        if delay > 1:
            model = AudioRNN(cell_type=DelayLineLSTMCell,
                             hidden_size=model_info['hidden_size'],
                             cell_args={'delay': delay})
        else:
            model = AudioRNN(cell_type=nn.LSTMCell,
                             hidden_size=model_info['hidden_size'])

    _, out_sig = model.apply({'params': params}, model.initialise_carry((1, 1)), in_sig)
    return jnp.ravel(out_sig)


def srirnn_inference(model_info: Dict, params: Dict,
                               in_sig: jax.Array,
                               orig_freq: int,
                               new_freq: int,
                               order: int,
                               method='lagrange') -> jax.Array:
    ratio = np.array([new_freq]) / np.array([orig_freq])

    if in_sig.ndim == 2:
        in_sig = jnp.expand_dims(in_sig, 0)
    elif in_sig.ndim == 1:
        in_sig = jnp.expand_dims(in_sig, (0, 2))

    cell = FIRInterpLSTMCell if model_info['unit_type'].lower() == 'lstm' else FIRInterpRNNCell
    delay = model_info['delay']
    delta = delay*ratio - 1

    # from "splitting the unit delay"
    if order % 2 == 0:
        skip = np.round(delta) - order//2
    else:
        skip = np.fix(delta) - (order-1)//2
    skip = 0 if skip < 0 else int(skip)
    delta -= skip
    kernel = utils.get_fir_interp_kernel(order, delta, method=method)
    kernel = kernel / np.sum(kernel)

    model_up = AudioRNN(cell_type=cell,
                        hidden_size=model_info['hidden_size'],
                        cell_args={'kernel': kernel, 'skip': skip}, dtype=jnp.float64)
    _, out_sig = model_up.apply({'params': params}, model_up.initialise_carry((1, 1)), in_sig)
    return jnp.ravel(out_sig)
