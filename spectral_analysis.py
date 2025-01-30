import numpy as np
from scipy.signal import cheby1, lfilter
from scipy.signal.windows import chebwin
import matplotlib.pyplot as plt


def bandlimit_signal(sig, fs: int, f0: float, IS_SYM=False, APPLY_LP=False, PLOT_SIG=False, cheb_at=-240):
    '''
    Generate alias-free bandlimited harmonic signal. Adapted from:
    https://github.com/victorzheleznov/dafx24/blob/master/metrics/bandlimit_signal.m
    '''
    # filter input signal
    if APPLY_LP:
        # use the same filter as in decimate() for oversampled algorithms
        b, a = cheby1(8, 0.05, 0.8)
        sig = lfilter(b, a, sig)

    # truncate signal to remove transient and leave one-second fragment
    if len(sig) > fs:
        sig = sig[-fs:]
    else:
        print('Warning: Signal is too short for robust metrics calculation!')

    N = len(sig)

    # check for odd length
    if N % 2 != 0:
        print('Signal should have an even length after truncation!')
    assert N % 2 == 0, 'Signal should have an even length after truncation!'

    # calculate harmonics
    num_harmonics = int(np.floor(0.5 * fs / f0))
    if IS_SYM:
        harmonics = f0 * np.arange(1, 2 * num_harmonics + 1, 2)
    else:
        harmonics = f0 * np.arange(1, num_harmonics + 1)

    # Chebyshev window
    win = chebwin(N, cheb_at, sym=False)

    # calculate single-sided spectrum
    S = np.fft.rfft(win * sig, N)

    # adjust for scalloping loss (including window)
    bins_exact = (harmonics * N / fs)  # exact bins for harmonics
    bins = np.round(bins_exact).astype(int)  # discrete bins for harmonics
    d = bins_exact - bins  # difference between exact and discrete bins
    idx = np.arange(N)  # indexes for summation

    # DC bin
    dc = np.real(S[0]) / np.sum(win)

    # synthesize bandlimited signal
    t = idx / fs
    sig_lim = np.ones_like(t) * dc  # start with DC
    amps = []
    phases = []
    for i in range(len(bins)):
        # calculate adjusted FFT value at discrete bin
        tmp = S[bins[i]] / np.sum(win * np.exp(1j * 2 * np.pi * d[i] / N * idx))

        # extract magnitude and phase
        amp = 2 * np.abs(tmp)
        phase = np.angle(tmp)

        amps.append(amp)
        phases.append(phase)

        # synthesis signal
        sig_lim += amp * np.cos(2 * np.pi * harmonics[i] * t + phase)

    amps = np.stack(amps)
    phases = np.stack(phases)
    complex_amps = amps * np.exp(1j * phases)


    # vectorised -- slower sometimes!
    # complex_amps = S[bins] / np.sum(np.expand_dims(win, -1) * np.exp(1j * 2 * np.pi * d * np.expand_dims(idx, -1) / N), axis=0)
    # amp = np.abs(complex_amps)
    # phase = np.angle(complex_amps)
    # partials = 2 * amp * np.cos(2 * np.pi * harmonics * np.expand_dims(t, -1) + phase)
    # sig_lim = dc + np.sum(partials, -1)

    # calculate aliased components
    alias = sig - sig_lim

    # debug plots
    if PLOT_SIG:
        plt.figure()
        plt.plot(sig, color="#e6194b", label="Input signal")
        plt.plot(sig_lim, color="#3cb44b", label="Bandlimited signal")
        plt.plot(alias, color="#4363d8", label="Alias signal")
        plt.xlim([-100, len(sig) + 100])
        plt.xlabel('Samples [n]', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.legend()
        plt.title(f"Input signal at $f_0 = {f0:.2f}$ Hz", fontsize=14)
        plt.grid(True)
        plt.xlim([0, 5*fs/f0])
        plt.show()

    return sig, sig_lim, alias, complex_amps
