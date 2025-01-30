import numpy as np
from scipy.signal import freqz
from filters import HalfbandIIR, HalfbandFIR, CIC, HSF
from utils import lagrange_upfirdn_filter
#matplotlib.use('macosx')
import matplotlib.pyplot as plt

# DESIGN PARAMETERS __________________
sr_base = 44100         # base sample rate [Hz]
M = 8                   # expansion/decimation factor
sba = 120               # min stop-band attenuation [dB]
pbr = 0.5               # max pass-band ripple      [dB]
pb_edge = 16e3          # pass-band edge [Hz]
sb_edge = 28.1e3        # stop-band edge [Hz]
sr = M * sr_base        # design sample rate

# plotting params
linewidth= 1.0
worN = 512


# CASCADED HB DESIGNS __________________________
num_cascades = int(np.log2(M))
H1 = np.ones(worN, dtype='complex128')
H2 = np.ones(worN, dtype='complex128')

fir_remez = HalfbandFIR(sr_base*2, pb_edge, sba)
h_remez, _ = fir_remez.get_coeffs()

iir_hb = HalfbandIIR(sr_base*2, pb_edge, N=13)
b, a = iir_hb.get_coeffs()

for n in range(num_cascades):
    H1 = np.concatenate([H1] * 2, axis=0)
    H2 = np.concatenate([H2] * 2, axis=0)

    _, H_fir = freqz(h_remez, worN=(2 ** (n+1)) * worN, whole=True)
    H1 *= H_fir

    _, H_iir = freqz(b, a, worN=(2 ** (n+1)) * worN, whole=True)
    H2 *= H_iir


# EQ+LINTERP _________________________
h_linterp = lagrange_upfirdn_filter(1, M)
w, H3 = freqz(h_linterp, fs=sr, worN=M*worN, whole=True)
b, a = HSF(G=1.941, fc=pb_edge, fs=sr_base).get_coeffs()
z = np.exp(1j * 2 * np.pi * w/sr)
_, H_hsf = freqz(b, a, worN, whole=True)
H_hsf = np.concatenate([H_hsf] * M, axis=0)  # response post expander
H3 *= H_hsf

# CIC (+EQ) ________________________________
N = 6
cic = CIC(L=M, N=N, D=1)
b_comb, a_int = cic.get_coeffs()
_, H4 = freqz(b_comb, a_int, worN=M*worN, whole=True)
H4 = H4 ** N                                # cascade
b, a = HSF(G=12.71, g=3.817e-6, fc=pb_edge, fs=sr_base).get_coeffs()
_, H_hs2 = freqz(b, a, worN, whole=True)
H_hs2 = np.concatenate([H_hs2] * M, axis=0)  # commute back through decimator
H4 *= H_hs2

# Plotting
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[12, 3])
colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
labels = ['C-HB-IIR', 'C-HB-FIR', 'EQ-Linterp', 'CIC']
H_list = [H2, H1, H3, H4]

for i, H in enumerate(H_list):

    ax_idx = int(np.floor(i/2))

    ax[ax_idx, 0].plot(w/1000, 20 * np.log10(np.abs(H)), linewidth=linewidth, color=colors[i], label=labels[i])
    ax[ax_idx, 0].grid(True, which='both', alpha=0.25)
    ax[ax_idx, 0].set_ylabel('Magnitude [dB]')
    ax[ax_idx, 0].set_xlim([1, sr/1000/2])
    ax[ax_idx, 0].set_ylim([-200, 5])
    ax[ax_idx, 0].vlines([16, 28.1], ymin=-300, ymax=5, linestyles='--', colors='k', linewidth=0.5*linewidth)
    ax[ax_idx, 0].hlines(-120, xmin=0, xmax=sr, colors='k', linestyles='--', linewidth=0.5*linewidth)
    ax[ax_idx, 0].legend(loc='upper right')

    if i == 1:
        linestyle = '--'
    else:
        linestyle = '-'
    ax[ax_idx, 1].plot(w / 1000, 20 * np.log10(np.abs(H)), linewidth=linewidth, color=colors[i], linestyle='-')
    ax[ax_idx, 1].grid(True, which='both', alpha=0.25)
    ax[ax_idx, 1].ticklabel_format(axis='y', style='sci')
    ax[ax_idx, 1].set_xticks(np.arange(0, 12) * 2)
    ax[ax_idx, 1].vlines([pb_edge/1000, sb_edge/1000], ymin=-300, ymax=5, linestyles='--', colors='k', linewidth=0.5*linewidth)
    ax[ax_idx, 1].set_xlim([0, 22])
    ax[ax_idx, 1].set_ylim([-1, 1])
    ax[ax_idx, 1].hlines(-0.5, xmin=0, xmax=sr, colors='k', linestyles='--', linewidth=0.5*linewidth)
    ax[ax_idx, 1].hlines(0.5, xmin=0, xmax=sr, colors='k', linestyles='--', linewidth=0.5*linewidth)
    ax[ax_idx, 1].set_ylabel('Magnitude [dB]')


ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])
ax[1, 0].set_xlabel('Frequency [kHz]')
ax[1, 1].set_xlabel('Frequency [kHz]')
fig.subplots_adjust(wspace=0.2, hspace=0.15)
plt.savefig('figs/filter_analysis_M=8.pdf', bbox_inches='tight')
plt.show()