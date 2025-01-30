import warnings

import numpy as np
from typing import Dict, Tuple
import json
import jax.numpy as jnp
import scipy
from jax import Array, jit
import math
from jax.random import PRNGKey
import os
import jax
import flax.linen as nn
jax.config.update("jax_enable_x64", True)
from librosa import A_weighting
from argparse import Namespace
import matlab.engine
eng = matlab.engine.start_matlab()
os.getcwd()
print(os.path.dirname(__file__))


BASE_PATH, _ = os.path.split(os.path.realpath(__file__))

def wavread_float32(filename):
    sample_rate, audio = scipy.io.wavfile.read(filename)
    dtype = audio.dtype
    if dtype != np.float32:
        audio = np.float32(audio / np.iinfo(dtype).max)
    return sample_rate, audio

def listdir_nohidden(path):
    return sorted((f for f in os.listdir(path) if not f.startswith(".")), key=str.lower)

def model_info_from_json(filename: str, flax: bool = True) -> Tuple[Dict, Dict]:
    '''
    Get info and state dict from Proteus Tone Library model
    :param filename: .json file
    :return: hyper_data, state_dict
    '''
    with open(filename, 'r') as f:
        json_data = json.load(f)

    if 'model_data' in json_data:
        print('Loading Proteus weights')
        hyper_data = json_data["model_data"]
        path, filename = os.path.split(filename)
        hyper_data["name"] = filename.split('.json')[0]
        state_dict = json_data["state_dict"]
        if flax:
            if hyper_data['unit_type'] == 'LSTM':
                state_dict = audio_LSTM_params_from_state_dict(state_dict)
            elif hyper_data['unit_type'] == 'RNN':
                state_dict = audio_RNN_params_from_state_dict(state_dict)
            else:
                print('WARNING: unrecognized cell type!')
            return hyper_data, state_dict
    else:
        print('Loading AIDA-X weights')
        return model_info_from_aida(filename)




def model_info_from_aida(filename: str) -> Tuple[Dict, Dict]:
    with open(filename, 'r') as f:
        json_data = json.load(f)

    model_info = {
        'hidden_size': json_data['layers'][0]['shape'][-1],
        'unit_type': json_data['layers'][0]['type'],
        'delay': 1
    }

    layer_data = json_data['layers']

    # state_dict['rec.weight_ih_l0']
    for l, layer in enumerate(layer_data):
        for w, weight in enumerate(layer['weights']):
            layer_data[l]['weights'][w] = jnp.asarray(weight).T

    lstm, lin = layer_data

    W_hi, W_hf, W_hg, W_ho = jnp.split(lstm['weights'][1], 4)
    W_ii, W_if, W_ig, W_io = jnp.split(lstm['weights'][0], 4)
    bi, bf, bg, bo = jnp.split(lstm['weights'][2], 4)

    lstm_params = {
        'hi': {'kernel': W_hi.transpose(), 'bias': bi.transpose()},
        'hf': {'kernel': W_hf.transpose(), 'bias': bf.transpose()},
        'hg': {'kernel': W_hg.transpose(), 'bias': bg.transpose()},
        'ho': {'kernel': W_ho.transpose(), 'bias': bo.transpose()},
        'ii': {'kernel': W_ii.transpose()},
        'if': {'kernel': W_if.transpose()},
        'ig': {'kernel': W_ig.transpose()},
        'io': {'kernel': W_io.transpose()}
    }

    linear_params = {'kernel': lin['weights'][0].T,
                     'bias': lin['weights'][1]}

    params = {'rec': {'cell': lstm_params}, 'linear': linear_params}


    return model_info, params

def audio_LSTM_params_from_state_dict(state_dict: Dict) -> Dict:
    '''
    Converts Proteus Tone Library .json files to flax parameter Dict
    :param state_dict: Proteus format parameters
    :return: flax format parameters
    '''

    for key, value in state_dict.items():
        state_dict[key] = jnp.asarray(value)

    W_hi, W_hf, W_hg, W_ho = jnp.split(state_dict['rec.weight_hh_l0'], 4)
    W_ii, W_if, W_ig, W_io = jnp.split(state_dict['rec.weight_ih_l0'], 4)
    bi, bf, bg, bo = jnp.split(state_dict['rec.bias_hh_l0'] + state_dict['rec.bias_ih_l0'], 4)

    lstm_params = {
        'hi': {'kernel': W_hi.transpose(), 'bias': bi.transpose()},
        'hf': {'kernel': W_hf.transpose(), 'bias': bf.transpose()},
        'hg': {'kernel': W_hg.transpose(), 'bias': bg.transpose()},
        'ho': {'kernel': W_ho.transpose(), 'bias': bo.transpose()},
        'ii': {'kernel': W_ii.transpose()},
        'if': {'kernel': W_if.transpose()},
        'ig': {'kernel': W_ig.transpose()},
        'io': {'kernel': W_io.transpose()}
    }

    linear_params = {'kernel': state_dict['lin.weight'].transpose(),
                     'bias': state_dict['lin.bias']}

    return {'rec': {'cell': lstm_params},
            'linear': linear_params}

def audio_LSTM_params_from_state_dict_aida(state_dict: Dict) -> Dict:
    '''
    Converts AIDA-X .json files to flax parameter Dict
    :param state_dict: Proteus format parameters
    :return: flax format parameters
    '''

    for key, value in state_dict.items():
        state_dict[key] = jnp.asarray(value)

    W_hi, W_hf, W_hg, W_ho = jnp.split(state_dict['rec.weight_hh_l0'], 4)
    W_ii, W_if, W_ig, W_io = jnp.split(state_dict['rec.weight_ih_l0'], 4)
    bi, bf, bg, bo = jnp.split(state_dict['rec.bias_hh_l0'] + state_dict['rec.bias_ih_l0'], 4)

    lstm_params = {
        'hi': {'kernel': W_hi.transpose(), 'bias': bi.transpose()},
        'hf': {'kernel': W_hf.transpose(), 'bias': bf.transpose()},
        'hg': {'kernel': W_hg.transpose(), 'bias': bg.transpose()},
        'ho': {'kernel': W_ho.transpose(), 'bias': bo.transpose()},
        'ii': {'kernel': W_ii.transpose()},
        'if': {'kernel': W_if.transpose()},
        'ig': {'kernel': W_ig.transpose()},
        'io': {'kernel': W_io.transpose()}
    }

    linear_params = {'kernel': state_dict['lin.weight'].transpose(),
                     'bias': state_dict['lin.bias']}

    return {'rec': {'cell': lstm_params},
            'linear': linear_params}

def audio_RNN_params_from_state_dict(state_dict: Dict) -> Dict:

    for key, value in state_dict.items():
        state_dict[key] = jnp.asarray(value)

    rnn_params = {
        'h': {'kernel': state_dict['rec.weight_hh_l0'].transpose()},
        'i': {'kernel': state_dict['rec.weight_ih_l0'].transpose(),
              'bias': state_dict['rec.bias_hh_l0'].transpose() + state_dict['rec.bias_ih_l0'].transpose()}
    }

    linear_params = {'kernel': state_dict['lin.weight'].transpose(),
                     'bias': state_dict['lin.bias']}

    return {'rec': {'cell': rnn_params},
            'linear': linear_params}


def state_dict_from_RNN_params(params: Dict) -> Dict:
    rnn_params = params['rec']['cell']
    linear_params = params['linear']


    state_dict = {
        'rec.weight_hh_l0': np.array(rnn_params['h']['kernel'].transpose()),
        'rec.weight_ih_l0': np.array(rnn_params['i']['kernel'].transpose()),
        'rec.bias_hh_l0': np.array(0.5 * rnn_params['i']['bias'].transpose()).squeeze(),
        'rec.bias_ih_l0': np.array(0.5 * rnn_params['i']['bias'].transpose()).squeeze(),
        'lin.weight': np.array(linear_params['kernel']).transpose(),
        'lin.bias': np.array(linear_params['bias'])
    }

    return state_dict

def state_dict_from_audio_LSTM_params(params: Dict) -> Dict:
    lstm_params = params['rec']['cell']
    linear_params = params['linear']

    # Inverse of splitting and transposing in audio_LSTM_params_from_state_dict
    W_hi = np.array(lstm_params['hi']['kernel']).transpose()
    W_hf = np.array(lstm_params['hf']['kernel']).transpose()
    W_hg = np.array(lstm_params['hg']['kernel']).transpose()
    W_ho = np.array(lstm_params['ho']['kernel']).transpose()

    W_ii = np.array(lstm_params['ii']['kernel']).transpose()
    W_if = np.array(lstm_params['if']['kernel']).transpose()
    W_ig = np.array(lstm_params['ig']['kernel']).transpose()
    W_io = np.array(lstm_params['io']['kernel']).transpose()

    bi = np.array(lstm_params['hi']['bias']).transpose()
    bf = np.array(lstm_params['hf']['bias']).transpose()
    bg = np.array(lstm_params['hg']['bias']).transpose()
    bo = np.array(lstm_params['ho']['bias']).transpose()

    # Recombine the parameters
    rec_weight_hh_l0 = np.concatenate([W_hi, W_hf, W_hg, W_ho], axis=0)
    rec_weight_ih_l0 = np.concatenate([W_ii, W_if, W_ig, W_io], axis=0)
    rec_bias_hh_l0 = np.concatenate([0.5 * bi, 0.5 * bf, 0.5 * bg, 0.5 * bo], axis=0)
    rec_bias_ih_l0 = np.concatenate([0.5 * bi, 0.5 * bf, 0.5 * bg, 0.5 * bo], axis=0)

    state_dict = {
        'rec.weight_hh_l0': rec_weight_hh_l0,
        'rec.weight_ih_l0': rec_weight_ih_l0,
        'rec.bias_hh_l0': rec_bias_hh_l0,
        'rec.bias_ih_l0': rec_bias_ih_l0,  # recover original by reversing sum
        'lin.weight': np.array(linear_params['kernel']).transpose(),
        'lin.bias': np.array(linear_params['bias'])
    }

    return state_dict


def giant_fft_resample_jnp(x: Array, orig_freq: int, new_freq: int):

    if orig_freq == new_freq:
        return x

    # lengths
    n_in = x.shape[-1]
    m = 2 * math.ceil(n_in / 2 / orig_freq)  # fft zero-pad factor
    n_fft_orig = m * orig_freq
    n_fft_new = m * new_freq
    n_out = math.ceil(new_freq / orig_freq * n_in)

    # fft
    x_fft_og = jnp.fft.rfft(x, n_fft_orig)
    x_fft_new = jnp.zeros((x.shape[0], n_fft_new // 2 + 1), dtype=x_fft_og.dtype)

    if new_freq > orig_freq:
        # pad fft
        x_fft_new = x_fft_new.at[..., 0:n_fft_orig // 2].set(x_fft_og[..., 0:n_fft_orig // 2])
        x_fft_new = x_fft_new.at[..., n_fft_orig // 2].set(0.5 * x_fft_og[..., n_fft_orig // 2])
    else:
        # truncate fft
        x_fft_new = x_fft_new.at[..., 0:n_fft_new // 2].set(x_fft_og[..., 0:n_fft_new // 2])

    # ifft
    x_new = jnp.fft.irfft(x_fft_new)

    # truncate and scale
    return x_new[..., :n_out] * new_freq / orig_freq

def giant_fft_resample(x, orig_freq: int, new_freq: int, taper = True):

    ndim = x.ndim

    if x.ndim == 1:
        x = np.expand_dims(x, 0)

    if orig_freq == new_freq:
        return x

    # lengths
    n_in = x.shape[-1]
    m = 2 * math.ceil(n_in / 2 / orig_freq)  # fft zero-pad factor
    n_fft_orig = m * orig_freq
    n_fft_new = m * new_freq
    n_out = math.ceil(new_freq / orig_freq * n_in)

    # fft
    x_fft_og = np.fft.rfft(x, n_fft_orig)
    x_fft_new = np.zeros((x.shape[0], n_fft_new // 2 + 1), dtype=x_fft_og.dtype)

    if new_freq > orig_freq:
        # pad fft
        x_fft_new[..., 0:n_fft_orig // 2] = x_fft_og[..., 0:n_fft_orig // 2]
        x_fft_new[..., n_fft_orig // 2] = 0.5 * x_fft_og[..., n_fft_orig // 2]
    else:
        # truncate fft
        x_fft_new[..., 0:n_fft_new // 2] = x_fft_og[..., 0:n_fft_new // 2]

    if taper:
        n_taper_bins = int(0.1 * (n_fft_new//2 + 1))
        w = 1 - np.hanning(2*n_taper_bins + 1)[:n_taper_bins]
        w = np.concatenate((np.ones(n_fft_new//2 + 1 - n_taper_bins), w))
        x_fft_new *= w

    # ifft
    x_new = np.fft.irfft(x_fft_new)

    # truncate and scale
    x_new = x_new[..., :n_out] * new_freq / orig_freq

    return np.squeeze(x_new)

def fft_based_interp_kernel(order: int, delta: float):
    n = np.arange(0, order+1)
    N = order
    window = np.cos(np.pi * (n - delta) / N) / np.sinc((n-delta)/N)
    return np.sinc(n - delta) * window


def lagrange_interp_kernel(order: int, delta: float, pad: int = 0):
    '''
    Fractional delay filter coefficients using Lagrange design method

    :param order: order of interpolation
    :param delta: fractional delay
    :param pad: zero pad
    :return: filter coefficients
    '''
    if order == 1 and delta > 1.0:
        delta -= np.floor(delta)
    kernel = np.ones(order + 1, dtype=np.double)
    for n in range(order + 1):
        for k in range(order + 1):
            if k != n:
                kernel[n] *= (delta - k) / (n - k)
    kernel = kernel / np.sum(kernel)
    if pad > 0:
        kernel = np.pad(kernel, (0, pad))
    return kernel

def l_inf_optimal_kernel(order: int, delta: float, bw: float = 0.5) -> jax.Array:
    '''
    Fractional delay filter coefficients from LUT (minimax method)

    :param order: order of interpolation (0 to 10)
    :param delta: fractional delay  (48/44.1 - 1 OR 44.1/48 - 1)
    :param bw: bandwidth used for optimisation (0.5 OR 0.9)
    :return: filter coefficients
    '''
    # delta = np.round(delta, 3)
    # df = pd.read_csv(f'lookup_tables/L_inf_delta={delta}_bw={bw}.csv', header=None)
    # return df.values[order-1, :order+1]

    x = eng.frac_delay_soco(delta, float(order+1), float(4096), 0.5, True)
    x = np.stack(x).squeeze()
    x = x / np.sum(x)
    return x


def get_fir_interp_kernel(order: int, delta: float, method: str):
    '''
    Helper function to get fractional delay filter coefficients using Lagrange or Minmax
    :param order: order of interpolation
    :param delta: fractional delay
    :param method: 'lagrange' or 'minimax' or 'naive'
    :return: filter coefficients
    '''
    if order == 0 or delta == 0.0:
        return np.ones(1)

    if method == 'lagrange':
        return lagrange_interp_kernel(order, delta)
    elif method == 'minimax' or method == 'optimal':
        return l_inf_optimal_kernel(order, delta)
    elif method == 'naive':
        return np.ones(1)
    else:
        print('Invalid interpolation method')
        return

def jnp_lagrange_interp_kernel(key: PRNGKey, order: int, delta: float, pad: int = 0):
    kernel = np.ones(order + 1)
    for n in range(order + 1):
        for k in range(order + 1):
            if k != n:
                kernel[n] *= (delta - k) / (n - k)
    if pad > 0:
        kernel = np.pad(kernel, (0, pad))
    return jnp.array(kernel, dtype=jnp.float64)

def jnp_kronecker_delta(key: PRNGKey, order: int):
    kernel = jnp.zeros(order + 1)
    kernel = kernel.at[0].set(1.0)
    return kernel

def jnp_optimal_interp_kernel_44to48k(key: PRNGKey, order: int, pad: int = 0):
    kernel = np.load(os.path.join(BASE_PATH, 'lookup_tables/optimal_fir_coeffs_44.1kto48k_DCnorm.npy'))
    kernel = kernel[order-1, :order+1]
    kernel = kernel / np.sum(kernel)
    if pad > 0:
        kernel = jnp.pad(kernel, (0, pad))
    return kernel


def jnp_optimal_interp_kernel_48to44k(key: PRNGKey, order: int, pad: int = 0):
    kernel = np.load(os.path.join(BASE_PATH, 'lookup_tables/optimal_fir_coeffs_48kto44.1k_DCnorm.npy'))
    kernel = kernel[order-1, :order+1]
    kernel = kernel / np.sum(kernel)
    if pad > 0:
        kernel = jnp.pad(kernel, (0, pad))
    return kernel

class ParamFreezer:
    def __init__(self, param_name: str):
        self.param_name = param_name

    def freeze(self, path, v):
        return 'frozen' if self.param_name in path else 'trainable'

    def freeze_all_except(self, path, v):
        return 'trainable' if self.param_name in path else 'frozen'


def merge_dicts(parent: Dict, child: Dict):
    '''
    Merge child into parent recursively (ChatGPT)
    '''
    for key in child:
        if key in parent:
            if isinstance(parent[key], dict) and isinstance(child[key], dict):
                # If both values are dictionaries, merge them recursively
                merge_dicts(parent[key], child[key])
            else:
                # Otherwise, overwrite the value in 'a' with the value from 'b'
                parent[key] = child[key]
        else:
            warnings.warn("child key not found in parent, will not be copied over: ", key)
    return parent



def cheb_fft(x: Array, at: float = -120) -> Array:
    win = scipy.signal.windows.chebwin(x.shape[-1], at=at, sym=False)
    return jnp.fft.rfft(x * win)


# half spec

def get_harmonics(complex_spec: Array, f0: int, sr: int):
    L = complex_spec.shape[0]
    fax = sr/2 * jnp.arange(0, L) / (L-1)
    spec_slice = slice(f0, L, f0)
    freqs = fax[spec_slice]
    amps = jnp.abs(complex_spec[spec_slice])
    phase = jnp.angle(complex_spec[spec_slice])
    dc_amp = jnp.real(complex_spec[0])
    return freqs, amps, phase, dc_amp
#
# def get_odd_harmonics(complex_spec: T, f0: int, sr: int):
#     L = complex_spec.shape[0]
#     fax = sr * torch.arange(0, L) / L
#     spec_slice = slice(f0, L//2, 2*f0)
#     freqs = fax[spec_slice]
#     amps = complex_spec[spec_slice].abs()
#     phase = complex_spec[spec_slice].angle()
#     dc_amp = torch.real(complex_spec[0])
#     return freqs, amps, phase, dc_amp
#
#
def bandlimited_harmonic_signal(freqs: Array, amps: Array, phase: Array, dc_amp: Array, t_ax: Array, sr: int) -> Array:

    amps = jnp.expand_dims(amps, 1)
    freqs = jnp.expand_dims(freqs, 1)
    phase = jnp.expand_dims(phase, 1)
    x = dc_amp + 2 * jnp.sum(amps * jnp.cos(2 * jnp.pi * jnp.squeeze(t_ax) * freqs + phase), 0)

    return x * 2 / sr
#
@jit
def snr_dB(sig: Array, noise: Array):
    snr = jnp.sum(sig**2) / jnp.sum(noise**2)
    return 10 * jnp.log10(snr)

def snr_A_weight_dB(sig: Array, noise: Array, freqs: Array):
    weights = 10 ** (A_weighting(freqs))
    #sig *= weights
    noise *= weights
    snr = jnp.sum(sig**2) / jnp.sum(noise**2)
    return 10 * jnp.log10(snr)

def log_spectral_distance(sig: Array, ref: Array):
    sig = np.log(np.abs(sig)**2)
    ref = np.log(np.abs(ref)**2)
    return np.sqrt(np.mean((sig - ref) ** 2, axis=0))


@jit
def complex_snr_dB(sig: Array, noise: Array):
    snr = jnp.sum(sig * jnp.conj(sig)) / jnp.sum(noise * jnp.conj(noise))
    return 10 * jnp.log10(jnp.real(snr))

def measure_aliasing(X, f0, sr, t_ax):
    # baseline bl harmonic sig
    freqs, amps, phase, dc_base = get_harmonics(X, f0, sr)
    out_sig_bl = bandlimited_harmonic_signal(freqs, amps, phase, dc_base, t_ax, sr)
    X_bl = cheb_fft(out_sig_bl)
    X_bl *= jnp.abs(X[f0]) / jnp.abs(X_bl[f0])
    return snr_dB(sig=jnp.abs(X_bl), noise=jnp.abs(X_bl) - jnp.abs(X))

def measure_thd(X, f0, sr):
    freqs, amps, phase, dc_base = get_harmonics(X, f0, sr)
    return jnp.sqrt(jnp.sum(amps[1:]**2)) / amps[0] / len(amps)


def get_srna_and_nmr(sig, sig_bl, f0, sr):

    MATLAB_PATH = f'{os.path.dirname(__file__)}/MATLAB/dafx24/metrics'
    if os.path.exists(MATLAB_PATH):
        eng.cd(MATLAB_PATH, nargout=0)
    else:
        warnings.warn('MATLAB path not set, skipping NMR calculation')

    Y_bl = cheb_fft(sig_bl)
    Y = cheb_fft(sig)
    f0_trunc = np.floor(f0).astype(int)
    peak_idx = f0_trunc + np.argmax(np.abs(Y[f0_trunc:f0_trunc+2]))
    gain_adjust = np.abs(Y[peak_idx]) / np.abs(Y_bl[peak_idx])

    Y_bl *= gain_adjust
    sig_bl *= gain_adjust
    if os.path.exists(MATLAB_PATH):
        nmr = eng.calc_nmr(np.expand_dims(np.array(sig), 1),
                           np.expand_dims(np.array(sig_bl), 1),
                           float(sr), float(64),
                           nargout=1)
    else:
        nmr = 0
    snra = snr_dB(sig=np.abs(Y_bl),
                        noise=np.abs(Y - Y_bl))

    return snra, nmr

def get_LSTM_fixed_point(lstm_params, cond_const=0.5, rand=True, method='empircal'):
    class LSTMCellOneState(nn.LSTMCell):
        def __call__(self, carry, inputs):
            new_carry, _ = super().__call__(carry, inputs)
            return new_carry, jnp.concatenate(new_carry, axis=-1)

    hidden_size = lstm_params['cell']['hi']['bias'].shape[0]
    input_size = lstm_params['cell']['ii']['kernel'].shape[0]
    rnn = nn.RNN(LSTMCellOneState(hidden_size))

    if rand:
        init_carry = 0.1 * (np.random.random((1, hidden_size)) - 0.5)
    else:
        init_carry = jnp.zeros((1, hidden_size))

    in_sig = jnp.zeros((1, 10000, 1))
    if input_size == 2:
        in_sig = jnp.concatenate((in_sig, cond_const * np.ones_like(in_sig)), axis=-1)

    if method == 'newton-raphson':

        @jit
        def forward_fn(current_state_vec):
            h, c = jnp.split(current_state_vec, 2, axis=-1)
            new_carry, _ = rnn.apply({'params': lstm_params},
                                     in_sig[:, 0:1, :],
                                     initial_carry=(h, c), return_carry=True)
            return jnp.concatenate(new_carry, axis=-1)

        x = jnp.concatenate((init_carry, init_carry), axis=-1)
        for i in range(1000):

            res = (x - forward_fn(x)).squeeze()
            jacobian = jnp.eye(2 * hidden_size) - jax.jacobian(forward_fn)(x).squeeze()
            step = jnp.linalg.solve(jacobian, res)

            damper = 1.0
            for sub_i in range(20):
                x_i = x - damper * step
                res_i = (x_i - forward_fn(x_i)).squeeze()
                if np.linalg.norm(res_i) > np.linalg.norm(res):
                    damper *= 0.5
                else:
                    x = x_i
                    break

            if np.linalg.norm(res) < 1e-9:
                break

        print(f'Final residiual = {np.linalg.norm(res)} after {i} iterations')
        fixed_point = x

    else:
        _, states = rnn.apply({'params': lstm_params},
                                   in_sig,
                                   initial_carry=(init_carry, init_carry),
                                   return_carry=True)

        fixed_point = np.mean(states[:, -1000:, :], axis=1)

    return fixed_point


def get_LSTM_jacobian(lstm_params, cond_const=0.5):
    hidden_size = lstm_params['cell']['hi']['bias'].shape[0]
    input_size = lstm_params['cell']['ii']['kernel'].shape[0]
    rnn = nn.RNN(nn.LSTMCell(hidden_size))

    in_sig = jnp.zeros((1, 1, 1))
    if input_size == 2:
        in_sig = jnp.concatenate((in_sig, cond_const * np.ones_like(in_sig)), axis=-1)

    fixed_point = get_LSTM_fixed_point(lstm_params, cond_const=cond_const, rand=False)
    @jit
    def forward_fn(current_state_vec):
        h, c = jnp.split(current_state_vec, 2, axis=-1)
        new_carry, _ = rnn.apply({'params': lstm_params},
                                 in_sig,
                                 initial_carry=(h, c), return_carry=True)
        return jnp.concatenate(new_carry, axis=-1)

    jacobian = jax.jacobian(forward_fn)(fixed_point).squeeze()

    return jacobian



def high_order_ss_to_first_order(fb_matrices):
    '''
    Converts TF of form H(z) = (I - \sum_{k=0}^{K-1} A_k z^{-k-1})^{-1} to form  H(z) = (I - z^{-1} \hat{A})^{-1},
    :param fb_matrices: (NxNxK) matrix where N is the state size and K is the filter order A_k
    :return: A_hat
    '''
    if fb_matrices.ndim == 1:
        fb_matrices = np.expand_dims(fb_matrices, (0, 1))

    og_state_size = fb_matrices.shape[0]
    assert (fb_matrices.shape[1] == og_state_size)
    filter_order = fb_matrices.shape[-1]

    if filter_order == 1:
        return fb_matrices.squeeze()
    A = np.concatenate(np.split(fb_matrices, filter_order, axis=-1), axis=1).squeeze(-1)
    lower = np.eye(A.shape[-1])[:-og_state_size, :]
    A = np.concatenate((A, lower), axis=0)
    return A

def namespace_to_dict(namespace):
    if isinstance(namespace, Namespace):
        return {key: namespace_to_dict(value) for key, value in vars(namespace).items()}
    else:
        return namespace

def lagrange_upfirdn_filter(order, L, num_taps=None):
    n = (np.arange(L) / L) + (order - 1) / 2
    p = np.ones((order + 1, L))
    for m in range(order + 1):
        for q in range(order + 1):
            if m != q:
                p[m, :] *= (n - q) / (m - q)

    if num_taps is not None:
        trunc = (order - num_taps + 1) // 2
        p = p[trunc:(p.shape[0] - trunc), :]
        p /= np.sum(p, axis=0)

    p = np.flip(p, axis=0)
    h = np.ravel(p)
    return h[1:] / L


def equiripple_ord(d, f_delta):
    return int(2 * (d - 13) / 14.36 / f_delta)

def upsample(x, L):
    shape = x.shape[-1] * L
    y = np.zeros(shape, dtype=x.dtype)
    y[::L] = x
    return y

def downsample(x, M):
    return x[::M]


if __name__ == '__main__':
    model_info, params = model_info_from_json('Proteus_Tone_Packs/AmpPack1/6505Plus_Red_DirectOut.json')

    model_info, params = model_info_from_aida('AIDA-X/140923_burning_sunn_LSTM-40.json')
