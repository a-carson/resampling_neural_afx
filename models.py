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

class LPFLSTMCell(nn.LSTMCell):
    b: Array = None
    a: Array = None
    skip: int = 0

    @compact
    def __call__(self, carry, inputs):
        c, h, c_apf, h_apf = carry
        new_c_apf = self.b[0] * c[..., self.skip] + self.b[1] * c[..., self.skip+1] - self.a[-1] * c_apf
        new_h_apf = self.b[0] * h[..., self.skip] + self.b[1] * h[..., self.skip+1] - self.a[-1] * h_apf
        (latest_c, latest_h), _ = super().__call__((new_c_apf, new_h_apf), inputs)
        new_c = jnp.roll(c.at[..., -1].set(latest_c), shift=1, axis=-1)
        new_h = jnp.roll(h.at[..., -1].set(latest_h), shift=1, axis=-1)
        return (new_c, new_h, new_c_apf, new_h_apf), latest_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Tuple[Array, Array, Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        mem_shape = batch_dims + (self.features, 2 + self.skip)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        c_apf = self.carry_init(key2, batch_dims + (self.features, ), self.param_dtype)
        h_apf = self.carry_init(key2, batch_dims + (self.features, ), self.param_dtype)
        return (c, h, c_apf, h_apf)

class APDLLSTMCell(nn.LSTMCell):
    delta: float = 1.0
    skip: int = 0

    @compact
    def __call__(self, carry, inputs):
        coeff = (1 - self.delta) / (1 + self.delta)
        c, h, c_apf, h_apf = carry
        new_c_apf = coeff * (c[..., self.skip] - c_apf) + c[..., self.skip+1]
        new_h_apf = coeff * (h[..., self.skip] - h_apf) + h[..., self.skip+1]
        (latest_c, latest_h), _ = super().__call__((new_c_apf, new_h_apf), inputs)
        new_c = jnp.roll(c.at[..., -1].set(latest_c), shift=1, axis=-1)
        new_h = jnp.roll(h.at[..., -1].set(latest_h), shift=1, axis=-1)
        return (new_c, new_h, new_c_apf, new_h_apf), latest_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Tuple[Array, Array, Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        mem_shape = batch_dims + (self.features, 2 + self.skip)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        c_apf = self.carry_init(key2, batch_dims + (self.features, ), self.param_dtype)
        h_apf = self.carry_init(key2, batch_dims + (self.features, ), self.param_dtype)
        return (c, h, c_apf, h_apf)

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


class LearnableFIRInterpLSTMCell(nn.LSTMCell):
    order: int = 0
    kernel_pad: int = 0
    orig_freq: int = 1
    new_freq: int = 1
    interp_kernel_init: str = 'lagrange'

    def setup(self) -> None:
        delta = self.new_freq / self.orig_freq - 1.0
        if self.interp_kernel_init == 'lagrange':
            self.interp_kernel = self.param('interp_kernel', utils.jnp_lagrange_interp_kernel, self.order, delta, self.kernel_pad)
        elif self.interp_kernel_init == 'delta':
            self.interp_kernel = self.param('interp_kernel', utils.jnp_kronecker_delta, self.order)
        elif self.interp_kernel_init == 'rand':
            self.interp_kernel = self.param('interp_kernel', nn.initializers.uniform(scale=1.0), (self.order+1,))
        elif self.interp_kernel_init == 'optimal_44to48k':
            self.interp_kernel = self.param('interp_kernel', utils.jnp_optimal_interp_kernel_44to48k, self.order, self.kernel_pad)
        elif self.interp_kernel_init == 'optimal_48to44k':
            self.interp_kernel = self.param('interp_kernel', utils.jnp_optimal_interp_kernel_48to44k, self.order, self.kernel_pad)
        else:
            print('Unrecognized interp_kernel_init')




    @compact
    def __call__(self, carry, inputs):
        c, h = carry
        k = self.interp_kernel / jnp.sum(self.interp_kernel)
        c_interp = jnp.dot(c, k)
        h_interp = jnp.dot(h, k)
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
        num_coeffs = self.order + self.kernel_pad + 1
        mem_shape = batch_dims + (self.features, num_coeffs)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)


class STNLSTMCell(nn.LSTMCell):
    kernel: Array = None

    @compact
    def __call__(self, carry, inputs):
        c, h = carry
        (temp_c, temp_h), _ = super().__call__((c[..., 0], h[..., 0]), inputs)
        c_concat = jnp.concatenate((jnp.expand_dims(temp_c, -1), c), -1)
        h_concat = jnp.concatenate((jnp.expand_dims(temp_h, -1), h), -1)
        k = self.kernel / jnp.sum(self.kernel)
        latest_c = jnp.dot(c_concat, k)
        latest_h = jnp.dot(h_concat, k)
        new_c = c.at[..., -1].set(latest_c)
        new_h = h.at[..., -1].set(latest_h)
        new_c = jnp.roll(new_c, shift=1, axis=-1)
        new_h = jnp.roll(new_h, shift=1, axis=-1)
        return (new_c, new_h), latest_h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = random.split(rng)
        num_coeffs = jnp.size(self.kernel) - 1
        mem_shape = batch_dims + (self.features, num_coeffs)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)


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

def apdl_inference(model_info: Dict, params: Dict,
                               in_sig: jax.Array,
                               orig_freq: int,
                               new_freq: int) -> jax.Array:
    if in_sig.ndim == 2:
        in_sig = jnp.expand_dims(in_sig, 0)
    elif in_sig.ndim == 1:
        in_sig = jnp.expand_dims(in_sig, (0, 2))

    delay = model_info['delay']
    delta = delay * new_freq / orig_freq - 1.0
    skip = 0
    while delta > 1.10:
        delta -= 1.0
        skip += 1

    model_up = AudioRNN(cell_type=APDLLSTMCell, hidden_size=model_info['hidden_size'], cell_args={'delta': delta, 'skip': skip})
    _, out_sig = model_up.apply({'params': params}, model_up.initialise_carry((1, 1)), in_sig)
    return jnp.ravel(out_sig)

def oversampled_inference_STN(params: Dict, in_sig: jax.Array, orig_freq: int, new_freq: int, order: int) -> jax.Array:
    model_up = AudioRNN(cell_type=STNLSTMCell, hidden_size=jnp.size(params['linear']['kernel'], 0),
                        cell_args={'kernel': utils.lagrange_interp_kernel(order, 1.0 - orig_freq / new_freq)})

    _, out_sig = model_up.apply({'params': params},
                                   model_up.initialise_carry((1, 1)),
                                   in_sig)
    return jnp.ravel(out_sig)


def weight_adjust_inference(model_info: Dict, params: Dict, in_sig: jax.Array, orig_freq: int, new_freq: int) -> jax.Array:

    ratio = new_freq/orig_freq
    size = model_info['hidden_size']

    if model_info['unit_type'] == 'RNN':
        gains = {'': 1.0}
    else:
        gains = {'i': 0.25,
                 'f': 0.25,
                 'g': 1.0,
                 'o': 0.25}

    for key, gain in gains.items():
        A = gain * params['rec']['cell']['h' + key]['kernel']
        eigs, vecs = np.linalg.eig(A)
        r_new = np.exp(np.log(np.abs(eigs))/ratio)
        angle_new = []
        for eig in eigs:
            if np.imag(eig) == 0:
                if np.real(eig) > 0:
                    angle_new.append(0.0)
                else:
                    angle_new.append(np.pi)
            else:
                angle_new.append(np.angle(eig) / ratio)

        angle_new = np.stack(angle_new)
        eigs_new = r_new * np.exp(1j * angle_new)
        A_new = vecs @ np.diag(eigs_new) @ np.linalg.inv(vecs)
        params['rec']['cell']['h' + key]['kernel'] = np.real(A_new) / gain

        in_adjust = (np.eye(size) - A_new.T) @ np.linalg.inv(np.eye(size) - A.T)

        b = params['rec']['cell']['i' + key]['kernel']
        b_new = in_adjust @ b.T
        params['rec']['cell']['i' + key]['kernel'] = np.real(b_new.T)


    return audio_rnn_inference(model_info, params, in_sig)
