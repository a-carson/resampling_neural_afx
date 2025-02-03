import fractions
import shutil
import sys
import time

import utils
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from os.path import join
from argparse import ArgumentParser
from glob import glob
import scipy
from resampling_exp_config import config
import spectral_analysis
from models import audio_rnn_inference, srirnn_inference
from scipy.signal import resample, lfilter
from utils import model_info_from_json, wavread_float32

def run_experiment(model_filename: str,
                      cond_const: float = 0.5,
                      in_type='audio',
                      sr_model: int = 44100,
                      sr_input: int = 44100,
                      f0=1e3,
                      dur=1.5,
                      max_amp=0.1,
                      methods_cfg=None):

    metrics = {}

    if in_type == 'audio':
        sr_model, in_sig_base = wavread_float32('../../../audio_datasets/dist_fx_192k/44k/test/input_32.wav')
        if sr_input != sr_model:
            new_num_samples = int(in_sig_base.shape[-1] * sr_input / sr_model)
            in_sig = resample(in_sig_base, new_num_samples)
        else:
            in_sig = in_sig_base
        trunc = 10000
    elif in_type == 'chirp':
        dur = 5
        lead_in = 0.5
        in_sig_base = scipy.signal.chirp(np.arange(0, dur*sr_model) / sr_model, f0=20, f1=5e3, t1=dur, method='linear')
        in_sig_base = 0.01 * np.concatenate((np.ones(int(lead_in*in_sig_base)), in_sig_base))
        if sr_input != sr_model:
            new_num_samples = int(in_sig_base.shape[-1] * sr_input / sr_model)
            in_sig = resample(in_sig_base, new_num_samples)
        else:
            in_sig = in_sig_base
    elif in_type == 'sine':
        t_ax_base = np.arange(0, dur*sr_model) / sr_model
        in_sig_base = max_amp * np.sin(2*np.pi*f0*t_ax_base)

        t_ax = np.arange(0, dur*sr_input) / sr_input
        in_sig = max_amp * np.sin(2*np.pi*f0*t_ax)
    else:
        return


    # model info
    model_info, params = model_info_from_json(model_filename)
    if model_info['unit_type'] == 'LSTM':
        in_channels = params['rec']['cell']['ii']['kernel'].shape[0]
    else:
        in_channels = 1
    if in_channels == 2:
        in_sig = np.stack((in_sig, cond_const * np.ones_like(in_sig)), -1)
        in_sig_base = np.stack((in_sig_base, cond_const * np.ones_like(in_sig_base)), -1)


    # baseline --------------------------------------------
    out_base = audio_rnn_inference(model_info, params, in_sig_base)
    out_base = lfilter(0.9 * np.array([1, 1]), np.array([1, 0.8]), out_base)

    out_sigs = []
    if in_type == 'sine':

        out_base_trunc, out_base_bl, aliases_base, amps_base = spectral_analysis.bandlimit_signal(out_base, sr_model, f0)
        snra, nmr = utils.get_srna_and_nmr(out_base_trunc, out_base_bl, f0, sr_model)
        metrics['base'] = {
            'ASR': -snra,
            'NMR': nmr,
        }
        out_sigs.append(out_base)
        print(metrics['base'])


    for cfg in methods_cfg:
        resampler_input = cfg['input']
        resampler_output = cfg['output']


        # input resampling
        if resampler_input == 'sine_synth':
            sr = sr_model * cfg['model_L']
            t_ax_up = np.arange(0, dur * sr) / sr
            in_sig_up = max_amp * np.sin(2*np.pi*f0*t_ax_up)
        elif resampler_input is None:
            in_sig_up = in_sig
        else:
            in_sig_up = resampler_input.forward(in_sig)

        # X = utils.cheb_fft(in_sig_up)
        # X = X / X.shape[0]
        # fvec = np.fft.rfftfreq(in_sig_up.shape[0], d=1/44100)
        # plt.plot(fvec, 20 * np.log10(np.abs(X)))
        # plt.title(in_sig_up)
        # plt.xlim([0, 44100])
        # plt.ylim([-120, 10])
        # plt.show()

        # RNN Processing
        if 'model_L' in cfg and 'model_M' in cfg:
            new_freq = cfg['model_L']
            orig_freq = cfg['model_M']
            order = cfg['model_order'] if 'model_order' in cfg else 1
            out_sig = srirnn_inference(model_info, params, in_sig_up, orig_freq, new_freq, order, 'lagrange')
        else:
            out_sig = audio_rnn_inference(model_info, params, in_sig_up)

        # output resample
        if resampler_output is not None:
            out_sig = resampler_output.forward(out_sig)

        # resample back to 44.1kHz "perfectly"
        if sr_input != sr_model:
            out_sig = resample(out_sig, out_base.shape[-1])

        out_sig = lfilter(0.9 * np.array([1, 1]), np.array([1, 0.8]), out_sig)

        if in_type == 'sine':

            out_sig_trunc, out_bl, aliases, amps = spectral_analysis.bandlimit_signal(out_sig, sr_model, f0)
            snr = utils.snr_dB(sig=out_base_bl, noise=out_base_bl-out_bl)

            n_harmonics = min(len(amps_base), len(amps))
            snrh = utils.snr_dB(sig=np.abs(amps_base[:n_harmonics]), noise=np.abs(amps_base[:n_harmonics]) - np.abs(amps[:n_harmonics]))
            snra, nmr = utils.get_srna_and_nmr(out_sig_trunc, out_bl, f0, sr_model)
            current_metrics = {
                'ESR': -snr,
                'MESR': -snrh,
                'ASR': -snra,
                'NMR': nmr,
            }

        else:
            #scipy.io.wavfile.write(f'{mode}.wav', sr, np.array(out_sig))
            snr = utils.snr_dB(sig=out_sig[trunc:], noise=out_sig[trunc:] - out_base[trunc:])
            current_metrics = {
                'SNR-audio': snr,
            }

        metrics[cfg['name']] = current_metrics
        out_sigs.append(out_sig)
        print({'name': cfg['name']} | current_metrics)

    return metrics, out_sigs


parser = ArgumentParser()
parser.add_argument('--log_results', action='store_true', help='Save results as csv')
parser.add_argument('--exp_no', type=int, default=0, help=
            '0: Training rate Fs = 44.1kHz, model rate Fs = 48kHz (Sec. V-A in paper). '
            '1: Training rate Fs = 48kHz, model rate Fs = 44.1kHz (Sec. V-B in paper). '
            '2: Training rate Fs = 44.1kHz, oversampling 8x, comparison against baseline Linterp+CIC method. (Sec. VI-B in paper). '
            '3: As 2) but with cross correlation experiment (see Table III in paper). '
            '4: Training rate Fs = 44.1kHz, oversampling by a factor of L (set through override_L) (Sec. VI-C in paper). ')
parser.add_argument('-L', type=int, default=-1)
parser.add_argument('--model_paths', type=str, default='Proteus_Tone_Packs/Selection/*.json')

if __name__ == '__main__':
    args = parser.parse_args()
    log = args.log_results


    # Shared config -----------------------------
    sine_amp = 0.1
    midi = np.arange(21, 109)
    f0s = 440 * 2 ** ((midi - 69)/12)


    if args.L > -1:
        override_L = args.L
    else:
        override_L = None

    # get experiment config
    cfg = config(args.exp_no, override_L=override_L)
    L = cfg['L']
    M = cfg['M']
    src_ratio = L / M



    # Create save dir -----------------------------------
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_time_secs = time.time()
    if log:
        exp_dir = join(os.path.dirname(__file__), 'results/', f'L={L}_M={M}_' + start_time)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copyfile(__file__, join(exp_dir, 'script.py'))


    # Get model details --------------------------------
    model_filenames = sorted(glob(args.model_paths))


    for model_filename in model_filenames:
        model_name = os.path.split(model_filename)[-1].split('.json')[0]
        print(model_name)

        if log:
            model_dir = join(exp_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

        # Create DFs ----------------------------------------
        num_rows = len(cfg['methods'])
        shared_idx = index=['ESR', 'ASR', 'MESR', 'NMR', 'SNR-audio']
        dfs = {'base': pd.DataFrame(index=shared_idx)}
        for method in cfg['methods']:
            dfs[method['name']] = pd.DataFrame(index=shared_idx)

        # audio SNR ---------------
        # out_metrics, out_sigs = run_experiment(model_filename=model_filename,
        #                                          in_type='audio',
        #                                          sr_model=sr_base,
        #                                          sr_input=sr_input,
        #                                          methods_cfg=methods)
        #
        # for m, out_sig in zip(out_metrics, out_sigs):
        #     scipy.io.wavfile.write(f'audio/{model_name}_{m}_audio.wav', sr_base, np.array(out_sig))
        #
        #print(out_metrics)

        if log:
            #scipy.io.wavfile.write(join(model_dir, method['name'] + '.wav'), int(L/M*sr_base), np.array(out_sig))
            for key, df in dfs.items():
                df.to_csv(join(model_dir, f'{key}.csv'))

        # loop through f0s ---------------------------------------------
        for f0 in f0s:
            print(f'f0 = {f0}')

            out_metrics, out_sigs = run_experiment(model_filename=model_filename,
                                                     in_type='sine',
                                                     sr_model=cfg['sr_base'],
                                                     sr_input=cfg['sr_input'],
                                                     max_amp=sine_amp,
                                                     f0=np.array(f0, dtype=np.double),
                                                     methods_cfg=cfg['methods'])

            # for method, out_sig in zip(out_metrics, out_sigs):
            #     X = utils.cheb_fft(out_sig)
            #     #X = X / X.shape[0]
            #     #X /= np.max(np.abs(X))
            #     fvec = np.fft.rfftfreq(out_sig.shape[0], d=1/44100)
            #     plt.figure()
            #     plt.semilogx(fvec, 20 * np.log10(np.abs(X)))
            #     plt.title(method)
            #     plt.xlim([10, 22050])
            #     plt.ylim([-80, 80])
            #     plt.show()

            # for m, out_sig in zip(out_metrics, out_sigs):
            #     scipy.io.wavfile.write(f'audio/{model_name}_L={L}_M={M}_{m}_{int(f0)}Hz.wav', sr_base, np.array(out_sig))

            for method_name, method_metrics in out_metrics.items():
                for metric, value in method_metrics.items():
                    dfs[method_name].at[metric, f'{np.round(f0, 3)}'] = value

            if log:
                for method_key, df in dfs.items():
                    df.to_csv(join(model_dir, f'{method_key}.csv'))

    print(time.time() - start_time_secs)





