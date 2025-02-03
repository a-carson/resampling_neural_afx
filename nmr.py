import time
from numba import jit
import numpy as np
from scipy.signal.windows import hann

'''
Noise to mask ratio computation class and helper functions, 
adapted into Python from:  https://github.com/victorzheleznov/dafx24/blob/master/metrics/calc_nmr.m

further references:
[1] BS.1387-2 (05/2023)
    Method for objective measurements of perceived audio quality
    https://www.itu.int/rec/R-REC-BS.1387
[2] Peter Kabal, "An examination and interpretation of ITU-R BS.1387:
    Perceptual evaluation of audio quality," Tech. Rep., Department of
    Electrical & Computer Engineering, McGill University, 2003
    report: https://mmsp.ece.mcgill.ca/Documents/Reports/2002/KabalR2002v2.pdf
    code: https://www-mmsp.ece.mcgill.ca/Documents/Downloads/PQevalAudio/PQevalAudio-v1r0.tar.gz
'''


@jit(nopython=True)
def ears_response(fs, N):
    '''
    calculate squared outer and middle ears frequency response (`PQWOME` from [2])
    :param fs: sample rate [Hz]
    :param N: FFT length
    :return: squared ears response
    '''
    f = (np.linspace(0, fs // 2, N // 2 + 1) / 1000)
    Adb = -2.184 * f ** (-0.8) + 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) - 0.001 * f ** (3.6)
    return 10 ** (Adb / 10)


@jit(nopython=True)
def critical_band_param():
    '''
    calculate parameters for critical bands (`PQCB` from [2])
    :return:
       Nc --- number of bands;
       fc --- center frequencies (column);
       fl --- lower frequency edges (column);
       fu --- upper frequency edges (column);
       dz --- band step in Bark scale.
    '''
    dz = 1 / 4
    fL = 80  # frequency bnd start[Hz]
    fU = 18000  # frequency band stop[Hz]

    # compute number of bands
    zL = 7 * np.arcsinh(fL / 650)
    zU = 7 * np.arcsinh(fU / 650)
    Nc = int(np.ceil((zU - zL) / dz))

    # frequency bands from BS.1387 - 2 standard[1, 2]
    fl = np.expand_dims(np.array([80.000, 103.445, 127.023, 150.762, 174.694,
                                  198.849, 223.257, 247.950, 272.959, 298.317,
                                  324.055, 350.207, 376.805, 403.884, 431.478,
                                  459.622, 488.353, 517.707, 547.721, 578.434,
                                  609.885, 642.114, 675.161, 709.071, 743.884,
                                  779.647, 816.404, 854.203, 893.091, 933.119,
                                  974.336, 1016.797, 1060.555, 1105.666, 1152.187,
                                  1200.178, 1249.700, 1300.816, 1353.592, 1408.094,
                                  1464.392, 1522.559, 1582.668, 1644.795, 1709.021,
                                  1775.427, 1844.098, 1915.121, 1988.587, 2064.590,
                                  2143.227, 2224.597, 2308.806, 2395.959, 2486.169,
                                  2579.551, 2676.223, 2776.309, 2879.937, 2987.238,
                                  3098.350, 3213.415, 3332.579, 3455.993, 3583.817,
                                  3716.212, 3853.817, 3995.399, 4142.547, 4294.979,
                                  4452.890, 4616.482, 4785.962, 4961.548, 5143.463,
                                  5331.939, 5527.217, 5729.545, 5939.183, 6156.396,
                                  6381.463, 6614.671, 6856.316, 7106.708, 7366.166,
                                  7635.020, 7913.614, 8202.302, 8501.454, 8811.450,
                                  9132.688, 9465.574, 9810.536, 10168.013, 10538.460,
                                  10922.351, 11320.175, 11732.438, 12159.670, 12602.412,
                                  13061.229, 13536.710, 14029.458, 14540.103, 15069.295,
                                  15617.710, 16186.049, 16775.035, 17385.420], dtype=np.double), -1)
    fc = np.expand_dims(np.array([91.708, 115.216, 138.870, 162.702, 186.742,
                                  211.019, 235.566, 260.413, 285.593, 311.136,
                                  337.077, 363.448, 390.282, 417.614, 445.479,
                                  473.912, 502.950, 532.629, 562.988, 594.065,
                                  625.899, 658.533, 692.006, 726.362, 761.644,
                                  797.898, 835.170, 873.508, 912.959, 953.576,
                                  995.408, 1038.511, 1082.938, 1128.746, 1175.995,
                                  1224.744, 1275.055, 1326.992, 1380.623, 1436.014,
                                  1493.237, 1552.366, 1613.474, 1676.641, 1741.946,
                                  1809.474, 1879.310, 1951.543, 2026.266, 2103.573,
                                  2183.564, 2266.340, 2352.008, 2440.675, 2532.456,
                                  2627.468, 2725.832, 2827.672, 2933.120, 3042.309,
                                  3155.379, 3272.475, 3393.745, 3519.344, 3649.432,
                                  3784.176, 3923.748, 4068.324, 4218.090, 4373.237,
                                  4533.963, 4700.473, 4872.978, 5051.700, 5236.866,
                                  5428.712, 5627.484, 5833.434, 6046.825, 6267.931,
                                  6497.031, 6734.420, 6980.399, 7235.284, 7499.397,
                                  7773.077, 8056.673, 8350.547, 8655.072, 8970.639,
                                  9297.648, 9636.520, 9987.683, 10351.586, 10728.695,
                                  11119.490, 11524.470, 11944.149, 12379.066, 12829.775,
                                  13294.850, 13780.887, 14282.503, 14802.338, 15341.057,
                                  15899.345, 16477.914, 17077.504, 17690.045], dtype=np.double), -1)
    fu = np.expand_dims(np.array([103.445, 127.023, 150.762, 174.694, 198.849,
                                  223.257, 247.950, 272.959, 298.317, 324.055,
                                  350.207, 376.805, 403.884, 431.478, 459.622,
                                  488.353, 517.707, 547.721, 578.434, 609.885,
                                  642.114, 675.161, 709.071, 743.884, 779.647,
                                  816.404, 854.203, 893.091, 933.113, 974.336,
                                  1016.797, 1060.555, 1105.666, 1152.187, 1200.178,
                                  1249.700, 1300.816, 1353.592, 1408.094, 1464.392,
                                  1522.559, 1582.668, 1644.795, 1709.021, 1775.427,
                                  1844.098, 1915.121, 1988.587, 2064.590, 2143.227,
                                  2224.597, 2308.806, 2395.959, 2486.169, 2579.551,
                                  2676.223, 2776.309, 2879.937, 2987.238, 3098.350,
                                  3213.415, 3332.579, 3455.993, 3583.817, 3716.212,
                                  3853.348, 3995.399, 4142.547, 4294.979, 4452.890,
                                  4643.482, 4785.962, 4961.548, 5143.463, 5331.939,
                                  5527.217, 5729.545, 5939.183, 6156.396, 6381.463,
                                  6614.671, 6856.316, 7106.708, 7366.166, 7635.020,
                                  7913.614, 8202.302, 8501.454, 8811.450, 9132.688,
                                  9465.574, 9810.536, 10168.013, 10538.460, 10922.351,
                                  11320.175, 11732.438, 12159.670, 12602.412, 13061.229,
                                  13536.710, 14029.458, 14540.103, 15069.295, 15617.710,
                                  16186.049, 16775.035, 17385.420, 18000.000], dtype=np.double), -1)

    return Nc, fc, fl, fu, dz


@jit(nopython=True)
def group_matrix(fs, N, fl, fu):
    '''
    calculate grouping matrix from FFT bins to critical bands
    :param fs: sample rate [Hz]
    :param N: FFT length
    :param fl: lower frequency edges
    :param fu: upper frequency edges
    :return: grouping matrix (number of bands x number of real FFT bins)
    '''
    df = fs / N
    idx = np.arange(0, N // 2 + 1)
    ku = (2 * idx + 1) / 2
    kl = (2 * idx - 1) / 2
    U = np.maximum(0, np.minimum(fu, ku * df) - np.maximum(fl, kl * df)) / df  # eq. (13) in [2]
    return U


@jit(nopython=True)
def internal_noise(fc):
    '''
    internal noise in the ear  (`PQIntNoise` from [2])
    :param fc: center band frequencies
    :return: internal noise value for each band
    '''
    f = fc / 1000  # [kHz]
    Edb = 1.456 * f ** (-0.8)
    return 10 ** (Edb / 10)


@jit(nopython=True)
def freq_spread_fast(E, Bs, fc, dz):
    '''
     calculate frequency spread effect across critical bands using recursive
     implementation (`PQ_SpreadCB` from [2])
    :param E: energy in critical bands ("pitch patterns" [2])
    :param Bs: normalising factor (calculated for E = 1 across all bands)
    :param fc: center band frequencies
    :param dz: Bark scale band step
    :return: energy in critical bands after frequency spreading effect ("unsmeared (in time) excitation patterns" [2])
    '''
    # number of bands
    Nc = len(E)

    # power law for addition of spreading
    e = 0.4

    # allocate storage
    aUCEe = np.zeros((Nc, 1))
    Ene = np.zeros((Nc, 1))
    Es = np.zeros((Nc, 1))

    # calculate energy dependent terms
    aL = 10 ** (-2.7 * dz)
    for m in range(Nc):
        aUC = 10 ** ((-2.4 - 23 / fc[m]) * dz)
        aUCE = aUC * E[m] ** (0.2 * dz)
        gIL = (1 - aL ** (m + 1)) / (1 - aL)
        gIU = (1 - aUCE ** (Nc - m)) / (1 - aUCE)
        En = E[m] / (gIL + gIU - 1)
        aUCEe[m] = aUCE ** e
        Ene[m] = En ** e

    # lower spreading
    Es[Nc - 1] = Ene[Nc - 1]
    aLe = aL ** e
    for m in range(Nc - 1, -1, -1):
        Es[m] = aLe * Es[m + 1] + Ene[m]

    # upper spreading i > m
    for m in range(Nc - 1):
        r = Ene[m]
        a = aUCEe[m]
        for i in range(m + 1, Nc):
            r = r * a
            Es[i] += r

    for i in range(Nc):
        Es[i] = Es[i] ** (1 / e) / Bs[i]

    return Es


@jit(nopython=True)
def peak_factor(fn, N):
    '''
    calculate peak factor (`PQ_gp` from [2])
    (since sinusoid can fall between FFt bins the largest bin value will be
    the peak factor times the peak of the continuous response)

    :param fn: normalised input frequency;
    :param N: frame length.
    :param W:
    :return: peak factor.
    '''
    # distance to the nearest DFT bin
    df = 1 / N
    k = np.floor(fn / df)
    dfN = np.minimum((k + 1) * df - fn, fn - k * df)

    dfW = dfN * (N - 1)
    return np.sin(np.pi * dfW) / (np.pi * dfW * (1 - dfW ** 2))


@jit(nopython=True)
def scaling_factor(N, Amax, fn, Lp):
    '''
    calculate sclaing for loudness and Hann window (`PQ_GL` from [2])

    :param N: frame length
    :param Amax: maximum amplitude
    :param fn: normalised frequency
    :param Lp: sound pressure level
    :return: scaling constant
    '''
    gp = peak_factor(fn, N)
    return 10 ** (Lp / 20) / (gp * Amax / 4 * (N - 1))


@jit(nopython=True)
def mask_offset(Nc, dz):
    '''
    calculate mask offset (`PQ_MaskOffset` from [2])

    :param Nc: number of bands
    :param dz: Bark scale band step
    :return: mask offset
    '''
    idx = np.arange(Nc)
    gm = 10 ** (-((idx <= 12 / dz) * 3 + (idx > 12 / dz) * (0.25 * idx * dz)) / 10)
    return np.expand_dims(gm, -1)


@jit(nopython=True)
def calc_nmr(sig, sig_ref, fs):
    '''

    :param sig: test signal
    :param sig_ref: reference signal
    :return: noise to mask ratio [NMR]
    :param sig: test signal (noisy)
    :param sig_ref: reference signal (noise free)
    :param fs: sample rate [Hz]
    :return:
    '''
    N = 2048
    overlap = 0.5
    ftest = 1019.5
    Lp = 92
    bit = 64

    # calculate STFT parameters
    win = np.hanning(N)
    Emin = 1e-12
    HA = int(np.round(N - overlap * N))  # hop size
    Amax = 2 ** (bit - 1)

    # outer and middle ear response
    w_sq = ears_response(fs, N)

    # critical bands parameters
    Nc, fc, fl, fu, dz = critical_band_param()
    fc = fc
    dz = dz
    U = group_matrix(fs, N, fl, fu)
    Nc = Nc

    # loudness calibration and window scaling compensation
    GL = scaling_factor(N, Amax, ftest / fs, Lp)

    # internal noise
    Ein = internal_noise(fc)

    # normalizing factor for frequency spread
    Bs = freq_spread_fast(np.ones((Nc, 1), dtype=np.double),
                          np.ones((Nc, 1), dtype=np.double), fc, dz)

    # mask offset
    gm = mask_offset(Nc, dz)

    # quantize audio to signed integer
    sig = np.round(sig * Amax)
    sig_ref = np.round(sig_ref * Amax)

    # loudness calibration and window scaling compensation
    sig = sig * GL
    sig_ref = sig_ref * GL

    # frames loop
    NF = int(np.floor((sig.shape[-1] - N) / HA))  # number of frames(truncating signal)
    Eb_err = np.zeros((Nc, NF), dtype=np.double)
    Es_ref = np.zeros((Nc, NF), dtype=np.double)

    for i in range(NF):
        # get current frame
        idx = np.arange(0, N) + i * HA
        X = np.fft.rfft(sig[idx] * win)
        X_ref = np.fft.rfft(sig_ref[idx] * win)

        # square magnitude
        X = np.abs(X) ** 2
        X_ref = np.abs(X_ref) ** 2

        # outer and middle ear filtering
        X *= w_sq
        X_ref *= w_sq

        # difference magnitude signal
        X_err = X - 2 * np.sqrt(X * X_ref) + X_ref

        # group into critical bands
        Eb_err[:, i] = np.maximum(U @ X_err, Emin)
        Eb_ref = np.maximum(U @ X_ref, Emin)

        # add internal noise
        E_ref = np.expand_dims(Eb_ref, -1) + Ein

        # frequency spreading
        Es_ref[:, i] = freq_spread_fast(E_ref, Bs, fc, dz)[:, 0]

    # calculate noise - to - mask
    M = gm * Es_ref
    nmr = Eb_err / M
    nmr_total = 10 * np.log10(np.mean(nmr))
    return nmr_total

