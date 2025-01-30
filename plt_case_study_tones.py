
from resampling_exp import run_experiment
import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from resampling_exp_config import config

parser = ArgumentParser()
parser.add_argument('--exp_no', type=int, default=0)
args = parser.parse_args()

# Shared config -----------------------------
sine_amp = 0.1
dur = 5.5

# unique config --------------------
if args.exp_no == 0:
    f0 = 27.5
    model_paths = 'Proteus_Tone_Packs/Selection/RockmanXPR_HighGain.json'
elif args.exp_no == 2:
    f0 = 4186
    model_paths = 'Proteus_Tone_Packs/Selection/MesaMiniRec_HighGain_DirectOut.json'
cfg = config(args.exp_no)
L = cfg['L']
M = cfg['M']
sr_input = cfg['sr_input']
# ------------------------------------------

device_name = os.path.split(model_paths)[-1].split('.')[0]
src_ratio = L / M
out_metrics, out_sigs = run_experiment(model_filename=model_paths,
                                         in_type='sine',
                                         sr_model=cfg['sr_base'],
                                         sr_input=cfg['sr_input'],
                                         max_amp=sine_amp,
                                         dur=dur,
                                         f0=f0,
                                         methods_cfg=cfg['methods'])

if args.exp_no == 0:
    ax_dict = {
        'base': 0,
        'naive': 0,
        'NB-Kaiser': 1,
        'HB-IIR+WB-Kaiser': 1,
        'NB-Remez': 2,
        'HB-IIR+WB-Remez': 2,
        'LIDL': 3,
        'CIDL': 3
    }
    plot_args = {
        'base': {
            'label': 'L=M=1',
            'linestyle': '-',
            'color': 'k'
        },
        'naive': {
            'label': 'naive',
            'linestyle': '--',
            'color': 'tab:grey'
        },
        'NB-Kaiser': {
            'label': 'NB-Kaiser',
            'linestyle': '-',
            'color': 'tab:blue'
        },
        'HB-IIR+WB-Kaiser': {
            'label': 'HB-IIR + WB-Kaiser',
            'linestyle': '--',
            'color': 'tab:green'
        },
        'NB-Remez': {
            'label': 'NB-Remez',
            'linestyle': '-',
            'color': 'tab:orange'
        },
        'HB-IIR+WB-Remez': {
            'label': 'HB-IIR + WB-Remez',
            'linestyle': '--',
            'color': 'tab:red'
        },
        'LIDL': {
            'label': 'LIDL',
            'linestyle': '-',
            'color': 'tab:purple'
        },
        'CIDL': {
            'label': 'CIDL',
            'linestyle': '--',
            'color': 'tab:brown'
        }
    }

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=[6, 6])
    for method, out_sig in zip(out_metrics, out_sigs):

        ax_idx = ax_dict[method]

        out_sig = out_sig[10000:]
        n_fft = out_sig.shape[0]
        X = np.fft.rfft(out_sig * np.blackman(len(out_sig)), n_fft)

        X = X / X.shape[0]
        X /= np.max(np.abs(X))
        fvec = np.fft.rfftfreq(n_fft, d=1 / 44100)

        num_harmonics = int(0.5 * sr_input / f0)
        harmonics = f0 * np.arange(1, num_harmonics + 1)
        X_dB = 20 * np.log10(np.abs(X))

        ax[ax_idx].semilogx(fvec, X_dB, linewidth=0.75, **plot_args[method])

        if ax_idx < 3:
            ax[ax_idx].set_xticklabels([])
        ax[ax_idx].set_xlim([10, 22.05e3])
        ax[ax_idx].set_ylim([-120, 0])
        ax[ax_idx].set_yticks(-np.flip(np.arange(0, 160, 40)))
        ax[ax_idx].set_ylabel('Mag. [dB]')

        ax[ax_idx].legend(loc='upper right', prop={'size': 8}, ncol=1)

    ax[-1].set_xlabel('Frequency [Hz]')
    fig.subplots_adjust(hspace=0.2)
    plt.savefig(f'figs/{device_name}_tones_L=160_M=147_single.pdf', bbox_inches='tight')

    plt.show()
elif args.exp_no == 2:

    ax_dict = {
        'base': 0,
        'EQ-Linterp + CIC': 1,
        'C-HB-FIR': 2,
        'C-HB-IIR': 3,
        'FFT': 4
    }

    plot_args = {
        'base': {
            'label': 'M=1 (no oversampling)',
            'color': 'k',
        },
        'EQ-Linterp + CIC': {
            'label': 'EQ-Linterp + CIC',
            'color': 'tab:red'
        },
        'C-HB-FIR': {
            'label': 'C-HB-FIR',
            'color': 'tab:green'
        },
        'C-HB-IIR': {
            'label': 'C-HB-IIR',
            'color': 'tab:orange'
        },
        'FFT': {
            'label': 'FFT',
            'color': 'tab:blue'
        },
    }

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=[6, 6])
    # ax[i].hlines(-120, xmin=0, xmax=22.05, colors='k', linestyles='--', linewidths=0.5)

    for method, out_sig in zip(out_metrics, out_sigs):

        if method not in ax_dict:
            continue
        i = ax_dict[method]
        out_sig = out_sig[22050:]
        N = len(out_sig)
        X = np.fft.rfft(out_sig * np.blackman(N))
        X = X / X.shape[0]
        X /= np.max(np.abs(X))
        fvec = np.fft.rfftfreq(out_sig.shape[0], d=1 / 44100)

        num_harmonics = int(0.5 * sr_input / f0)
        harmonics = f0 * np.arange(1, num_harmonics + 1)
        bins = harmonics * int(N/sr_input)
        X_dB = 20 * np.log10(np.abs(X))

        ax[i].plot(fvec / 1000, X_dB, linewidth=0.75, **plot_args[method])
        ax[i].scatter(harmonics / 1000, X_dB[bins], marker='x', s=8, color=plot_args[method]['color'])
        ax[i].set_xlim([0, 22.05])
        ax[i].set_ylim([-180, 0])

        ax[i].set_yticks(-np.flip(np.arange(0, 240, 60)))
        if i < 4:
            ax[i].set_xticklabels([])
        ax[i].set_xlabel('Frequency [kHz]')
        ax[i].set_ylabel('Mag. [dB]')

        ax[i].legend(loc='upper left', prop={'size': 8}, ncol=1)
    fig.subplots_adjust(hspace=0.2)

    plt.savefig(f'figs/{device_name}_tones_M=8.pdf', bbox_inches='tight')

    plt.show()

