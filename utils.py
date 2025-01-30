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



def cheb_fft(x: Array, at: float = -120) -> Array:
    win = scipy.signal.windows.chebwin(x.shape[-1], at=at, sym=False)
    return jnp.fft.rfft(x * win)


@jit
def snr_dB(sig: Array, noise: Array):
    snr = jnp.sum(sig**2) / jnp.sum(noise**2)
    return 10 * jnp.log10(snr)


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


