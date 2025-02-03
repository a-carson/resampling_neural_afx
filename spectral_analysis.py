import numpy as np
from scipy.signal import cheby1, lfilter
from scipy.signal.windows import chebwin
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def bandlimited_fourier_resynthesis(sig, bins, bins_exact, harmonics, win, fs):
    N = len(sig)
    # calculate single-sided spectrum
    S = np.fft.rfft(win * sig, N)

    # adjust for scalloping loss (including window)
    d = bins_exact - bins  # difference between exact and discrete bins
    idx = np.arange(N)  # indexes for summation

    # DC bin
    dc = np.real(S[0]) / np.sum(win)

    # synthesize bandlimited signal
    t = idx / fs
    sig_lim = np.ones_like(t) * dc  # start with DC
    amps = np.zeros_like(harmonics)
    phases = np.zeros_like(harmonics)
    for i in range(len(bins)):
        # calculate adjusted FFT value at discrete bin
        tmp = S[bins[i]] / np.sum(win * np.exp(1j * 2 * np.pi * d[i] / N * idx))

        # extract magnitude and phase
        amp = 2 * np.abs(tmp)
        phase = np.angle(tmp)

        amps[i] = amp
        phases[i] = phase

        # synthesis signal
        sig_lim += amp * np.cos(2 * np.pi * harmonics[i] * t + phase)

    complex_amps = amps * np.exp(1j * phases)

    return sig_lim, complex_amps

def bandlimit_signal(sig, fs: int, f0: float, IS_SYM=False, cheb_at=-240):
    '''
    Generate alias-free bandlimited harmonic signal. Adapted from:
    https://github.com/victorzheleznov/dafx24/blob/master/metrics/bandlimit_signal.m
    '''

    # truncate signal to remove transient and leave one-second fragment
    if len(sig) > fs:
        sig = sig[-fs:]
    else:
        print('Warning: Signal is too short for robust metrics calculation!')

    N = len(sig)

    # check for odd length
    assert N % 2 == 0, 'Signal should have an even length after truncation!'

    # calculate harmonics
    num_harmonics = int(np.floor(0.5 * fs / f0))
    if IS_SYM:
        harmonics = f0 * np.arange(1, 2 * num_harmonics + 1, 2)
    else:
        harmonics = f0 * np.arange(1, num_harmonics + 1)

    # Chebyshev window
    win = chebwin(N, cheb_at, sym=False)

    bins_exact = (harmonics * N / fs)  # exact bins for harmonics
    bins = np.round(bins_exact).astype(int)  # discrete bins for harmonics
    sig_lim, complex_amps = bandlimited_fourier_resynthesis(sig, bins, bins_exact, harmonics, win, fs)
    # calculate aliased components
    alias = sig - sig_lim


    return sig, sig_lim, alias, complex_amps