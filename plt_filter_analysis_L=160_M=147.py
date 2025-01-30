import os

import numpy as np
from scipy.signal import freqz
from filters import Remez, Kaiser, HalfbandIIR
#matplotlib.use('macosx')
import matplotlib.pyplot as plt

# DESIGN PARAMETERS __________________
sr_base = 44100                             # base sample rate [Hz]
L = 160                                     # expansion factor
sba = 120                                   # min stop-band attenuation [dB]
pbr = 0.5                                   # max pass-band ripple      [dB]
pb_edge = 16e3                              # pass-band edge            [Hz]
sb_edge_narrow = 28.1e3                     # stop-band edge for narrow-band filters [Hz]
sb_edge_wide = 2*sr_base - sb_edge_narrow   # stop-band edge for wider-band filters [Hz]
sr = L * sr_base                            # design sample rate

# plotting params
worN = 1024
linewidth= 1.2

# SINGLE-STAGE METHODS ____________________
# NB-Kaiser
nb_kaiser = Kaiser(sr, 11.5e3, sb_edge_narrow, sba)
h_nb_kaiser, _ = nb_kaiser.get_coeffs()
w, H_nb_kaiser = freqz(h_nb_kaiser, fs=sr, worN=L*worN, whole=True)

# NB-Remez
nb_remez = Remez(sr, pb_edge, sb_edge_narrow, pbr, sba)
h_nb_remez, _ = nb_remez.get_coeffs()
w, H_nb_remez = freqz(h_nb_remez, fs=sr, worN=L*worN, whole=True)


# TWO-STAGE METHODS ----------------------------
# WB-Kaiser
wb_kaiser = Kaiser(sr, 0, sb_edge_wide, sba)
h_wb_kaiser, _ = wb_kaiser.get_coeffs()
w, H_wb_kaiser = freqz(h_wb_kaiser, fs=sr, worN=L*worN, whole=True)

# WB-Remez
wb_remez = Remez(sr, pb_edge, sb_edge_wide, pbr, sba)
h_wb_remez, _ = wb_remez.get_coeffs()
w, H_wb_remez = freqz(h_wb_remez, fs=sr, worN=L*worN, whole=True)

# HB-IIR
hb_iir = HalfbandIIR(2*sr_base, pb_edge, N=13)
b, a = hb_iir.get_coeffs()
a1, a2 = hb_iir.get_apf_coeffs()
_, Hhb = freqz(b, a, fs=2*sr_base, worN=2*worN, whole=True)
Hhb = np.concatenate([Hhb] * (L//2), axis=0)

# Cascade
H_wb_kaiser *= Hhb
H_wb_remez *= Hhb

# Plotting
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[12, 3])
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
#colors = ['k'] * 4
labels = ['NB-Kaiser', 'HB-IIR + WB-Kaiser', 'NB-Remez', 'HB-IIR + WB-Remez']
H_list = [H_nb_kaiser, H_wb_kaiser, H_nb_remez, H_wb_remez]
for i, H in enumerate(H_list):
    
    ax_idx = int(np.floor(i/2))
    ax[ax_idx, 0].semilogx(w/1000, 20 * np.log10(np.abs(H)), linewidth=linewidth, color=colors[i], label=labels[i])
    ax[ax_idx, 0].grid(True, which='both', alpha=0.25)

    ax[ax_idx, 0].set_ylabel('Magnitude [dB]')
    ax[ax_idx, 0].set_xlim([1, sr/1000/2])
    ax[ax_idx, 0].set_ylim([-200, 5])
    ax[ax_idx, 0].vlines([pb_edge, sb_edge_narrow], ymin=-300, ymax=5, linestyles='--', colors='k', linewidth=0.5*linewidth)
    ax[ax_idx, 0].hlines(-120, xmin=0, xmax=sr, colors='k', linestyles='--', linewidth=0.5*linewidth)
    ax[ax_idx, 0].legend()

    ax[ax_idx, 1].plot(w / 1000, 20 * np.log10(np.abs(H)), linewidth=linewidth, color=colors[i])
    ax[ax_idx, 1].grid(True, which='both', alpha=0.25)
    ax[ax_idx, 1].set_xticks(np.arange(0, 9) * 2)
    ax[ax_idx, 1].vlines([pb_edge/1000, sb_edge_narrow/1000], ymin=-300, ymax=5, linestyles='--', colors='k', linewidth=0.5*linewidth)
    ax[ax_idx, 1].set_xlim([0, 16])
    ax[ax_idx, 1].set_ylim([-1, 1])
    ax[ax_idx, 1].hlines(-0.5, xmin=0, xmax=sr, colors='k', linestyles='--', linewidth=0.5*linewidth)
    ax[ax_idx, 1].hlines(0.5, xmin=0, xmax=sr, colors='k', linestyles='--', linewidth=0.5*linewidth)
    ax[ax_idx, 1].set_ylabel('Magnitude [dB]')


ax[0, 0].set_xticklabels([])
ax[0, 1].set_xticklabels([])
ax[1, 0].set_xlabel('Frequency [kHz]')
ax[1, 1].set_xlabel('Frequency [kHz]')
fig.subplots_adjust(wspace=0.2, hspace=0.15)
plt.savefig('figs/filter_analysis_L=160_M=147.pdf', bbox_inches='tight')
plt.show()