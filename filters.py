from scipy.signal import ellip, ellipord, ellipap, lfilter, \
    resample_poly, remez, kaiserord, firwin, firls, kaiser_atten, kaiser_beta, resample
import numpy as np
from numpy import convolve as conv
from typing import List
import utils
from utils import lagrange_upfirdn_filter


class Resampler():
    '''
    Resampler for pre and post audio RNN interpolation/decimation
    (if input is stereo, it assumes second channel is constant and therefore doesn't resample)
    '''
    def __init__(self, L, M, filter_class, filter_args, ideal=False, pre_filter=None, post_filter=None):
        if ideal is False:
            self.filter = filter_class(**filter_args)
        self.L = L
        self.M = M
        self.ideal = ideal

        self.pre_filter = pre_filter
        self.post_filter = post_filter

    def forward(self, x):
        n_channels = x.ndim

        if n_channels == 2:
            cond_channel = x[..., -1]
            cond_const = cond_channel[0]
            x = x[..., 0]

        if self.pre_filter is not None:
            x = self.pre_filter.forward(x)

        if self.ideal:
            new_num_samples = int(x.shape[-1] * self.L / self.M)
            y = resample(x, new_num_samples)
        else:
            b, a = self.filter.get_coeffs()

            if not isinstance(a, int):
                y = utils.upsample(x, self.L)
                y = self.L * self.filter.forward(y)
                y = utils.downsample(y, self.M)
            else:
                h, _ = self.filter.get_coeffs()
                y = resample_poly(x, up=self.L, down=self.M, window=h)


        if self.post_filter is not None:
            y = self.post_filter.forward(y)

        if n_channels == 2:
            y = np.stack((y, cond_const * np.ones_like(y)), -1)

        return y


class MultiStageResampler():
    '''
    Multi-stage resampler made up of a cascade of Resampler objects
    '''
    def __init__(self, resamplers: List[Resampler]):
        self.resamplers = resamplers

    def forward(self, x):
        for resampler in self.resamplers:
            x = resampler.forward(x)
        return x


class Kaiser():
    '''
    Kaiser window FIR filter
    '''
    def __init__(self, sr, pb_edge, sb_edge, ripple_sb=None, N=None):
        f_delta = (sb_edge - pb_edge) / (sr / 2)
        f_c = (sb_edge + pb_edge) / sr

        if N is not None:
            N += (N % 2)
            beta = kaiser_beta(kaiser_atten(N+1, f_delta))
        else:
            beta = kaiser_beta(ripple_sb)
            N = int(np.ceil((ripple_sb - 7.95) / (14.36 * f_delta * 0.5)))
            N += (N % 2)


        self.N = N
        self.h = firwin(N + 1, f_c, window=('kaiser', beta))


    def get_coeffs(self):
        return self.h, 1


class LS():
    '''
    Least-squares optimal FIR filter
    '''
    def __init__(self, sr, pb_edge, sb_edge, ripple_sb=None, N=None):
        f_delta = (sb_edge - pb_edge) / (sr / 2)

        if N is None:
            N, beta = kaiserord(ripple_sb, f_delta)

        N += (N % 2)
        self.N = N
        self.h = firls(N+1, [0, pb_edge, sb_edge, 0.5 * sr], [1, 1, 0, 0], fs=sr)
        self.h /= np.sum(self.h)

    def get_coeffs(self):
        return self.h, 1

class Remez():
    '''
    Equiripple FIR filter designed using the Remez method with order estimation using Belanger's formula

    P. P. Vaidyanathan, "Multirate Systems and Filter Banks". Englewood Cliffs, N.J,
            Prentice Hall, 1993
    '''
    def __init__(self, sr, pb_edge, sb_edge, ripple_pb, ripple_sb, N=None):
        f_delta = (sb_edge - pb_edge) / sr
        ds = 10 ** (-ripple_sb / 20)
        dp = (10 ** (ripple_pb / 20) - 1) / (10 ** (ripple_pb / 20) + 1)

        if N is None:
            N = 2 * np.log10(1 / (10 * dp * ds)) / 3 / f_delta      # Belanger formula

        N = int(np.round(N))
        N += (N % 2)
        self.N = N
        h = remez(N+1, [0, pb_edge, sb_edge, 0.5 * sr], [1, 0], fs=sr, weight=[1, dp/ds], maxiter=250)
        self.h = h / np.sum(h)


    def get_coeffs(self):
        return self.h, 1


class HalfbandFIR():
    '''
    Halfband equiripple FIR filter designed using the method in:
        P. Vaidyanathan and T. Nguyen, “A ‘trick’ for the design of FIR half-
        band filters,” IEEE Trans. Circuits and Systems, vol. 34, no. 3, pp. 297–
        300, 1987

    See also:
        https://tomverbeure.github.io/2020/12/15/Half-Band-Filters-A-Workhorse-of-Decimation-Filters.html
    '''
    def __init__(self, sr, pb_edge, ripple_sb, N=None):

        f_delta = 0.5 - 2 * pb_edge/sr

        if ripple_sb is None and N is None:
            raise Exception('Either ripple_sb or order must be set')

        if ripple_sb is not None and N is not None:
            print('Filter order specified, ignoring ripple_sb')

        if N is None:
            ds = 10 ** (-ripple_sb / 20)
            N = 2 * np.log10(1 / (10 * ds ** 2)) / 3 / f_delta      # Belanger formula to estimate order

        N = int(np.round(N))

        # make N even but not power of 4 -------
        N += (N % 2)
        if N % 4 == 0:
            N += 2
        self.N = N
        g = remez(N//2+1, [0, 2*pb_edge/sr, 0.5, 0.5], [1, 0], [1, 1], maxiter=250) # the "trick"
        h = utils.upsample(g, 2)[:-1] / 2       # period filter (odd taps only)
        h[N // 2] = 0.5                         # set center tap

        self.h = h / np.sum(h)                  # normalise


    def get_coeffs(self):
        return self.h, 1

class Lagrange():
    '''

    '''
    def __init__(self, N, L):
        self.N = N
        self.h = lagrange_upfirdn_filter(order=N, L=L)

    def get_coeffs(self):
        return self.h, 1


class Ellip():
    def __init__(self, sr, ripple_sb, pb_edge, sb_edge, ripple_pb=0.5):
        N, wn = ellipord(wp=pb_edge, ws=sb_edge, gpass=ripple_pb, gstop=ripple_sb, fs=sr)
        self.N = N
        b, a = ellip(N=N, Wn=2 * wn / sr, rp=ripple_pb, rs=ripple_sb)
        self.b = b
        self.a = a
        print(f'Ellip filter order = {N}')

    def forward(self, x):
        return lfilter(self.b, self.a, x)


    def get_coeffs(self):
        return self.b, self.a

class HalfbandIIR():
    '''
    Halfband power-symmetric elliptic IIR filter.

    Filter coefficients are derived using the method in Table 5.3.1 in [1]. Also available in [2].

        [1] P. P. Vaidyanathan, "Multirate Systems and Filter Banks". Englewood Cliffs, N.J,
            Prentice Hall, 1993
        [2] R. Valenzuela and A. Constantinides, “Digital signal processing schemes
            for efficient interpolation and decimation,” IEE Proc. G—Electronic
            Circuits and Systems, vol. 130, pp. 225 – 235, Jan. 1984. Available: https://www.researchgate.net/publication/224336631_Digital_signal_processing_schemes_for_efficient_interpolation_and_decimation

    '''
    def __init__(self, sr, pb_edge, ripple_sb=None, N=None):


        if ripple_sb is None and N is None:
            raise Exception('Either ripple_sb or order must be set')

        if ripple_sb is not None and N is not None:
            print('Filter order specified, ignoring ripple_sb')

        # band edges in radians
        wp = 2 * np.pi * np.double(pb_edge) / sr
        ws = np.pi - wp

        # Derived quantities
        r = np.tan(0.5 * wp) / np.tan(0.5 * ws)
        r_hat = np.sqrt(1 - r**2)
        q0 = 0.5 * (1 - np.sqrt(r_hat)) / (1 + np.sqrt(r_hat))
        q = q0 + 2*q0**5 + 15*q0**9 + 150*q0**13

        # Order estimation ---------------
        if N is None:
            delta2 = 10 ** (-ripple_sb / 20)
            D = ((1 - delta2**2) / delta2**2)**2
            N = np.floor(np.log10(16 * D) / np.log10(1 / q)).astype(int)
            N += (N % 2) + 1         # round to next odd integer
        elif N % 2 == 0:
            raise Exception('N must be odd')
        # ________________________________

        # Computing the filter coefficients ----------------------
        m = (N - 1) // 2
        Omega = []               # $\Omega_k$
        inf_sum = 9              # use 9 terms in expansion
        for k in range(1, m+1):
            num = np.zeros(1, dtype=np.double)
            denom = np.zeros(1, dtype=np.double)

            for i in range(inf_sum+1):
                num += (-1)**i * q**(i*(i + 1)) * np.sin((2 * i + 1) * k * np.pi / N)
                denom += (-1)**(i+1) * q**((i + 1)**2) * np.cos(2 * np.pi * k * (i + 1) / N)

            num *= 2 * q**0.25
            denom = 1 + 2 * denom
            Omega.append(num / denom)

        Omega = np.stack(Omega, axis=0)
        v = np.sqrt((1 - r * Omega**2) * (1 - Omega**2 / r))    # $v_k$
        b = 2 * v / (1 + Omega**2)                              # $b_k$
        alpha = (2 - b) / (2 + b)                               # coefficients of first-order APFs, $\alpha_k$
        alpha = np.sort(alpha).squeeze()

        # denominator coefficients of cascaded APFs
        d0 = np.ones(1, dtype=np.double)                        # $d_0$
        d1 = np.ones(1, dtype=np.double)                        # $d_1$
        for k in range(m):
            dd = np.array([1, alpha[k]], dtype=np.double)
            if (k % 2) == 0:
                d0 = conv(d0, dd)
            else:
                d1 = conv(d1, dd)

        # LPF coefficients from combined polyphase APFs ----------------
        d0up = utils.upsample(d0, 2)[:-1]
        d1up = utils.upsample(d1, 2)[:-1]
        b = 0.5 * (conv(np.array([1, 0]), conv(np.flip(d0up), d1up)) +
                   conv(np.array([0, 1]), conv(np.flip(d1up), d0up)))
        a = conv(d0up, d1up)

        # Readjusting ripple size (for info only) ----------------------
        D_new = (10 ** (N * np.log10(1 / q))) / 16
        delta2_new = np.sqrt(1 / (1 + np.sqrt(D_new)))
        ripple_sb_new = -20 * np.log10(delta2_new)

        self.b = b
        self.a = a
        self.d0 = d0
        self.d1 = d1
        self.N = len(self.b) - 1
        n0 = len(self.d0) - 1
        n1 = len(self.d1) - 1
        N = 2 * (n0 + n1) + 1
        print(f'Halfband IIR order = 2({n0} + {n1}) + 1 = {N}. SBA = {np.round(ripple_sb_new, 3)}dB')


    def forward(self, x):
        return lfilter(self.b, self.a, x)


    def get_coeffs(self):
        return self.b, self.a

    def get_apf_coeffs(self):
        return self.d0, self.d1

class CIC():
    '''
    Cascaded comb-integrator decimator

    The default values are those used in:
    J. Kahles, F. Esqueda, and V. Välimäki, “Oversampling for nonlinear
    waveshaping: Choosing the right filters,” J. Audio Eng. Soc., vol. 67,
    no. 6, pp. 440–449, Jun. 2019
    '''
    def __init__(self, L: int, N: int=6, D: int=1):
        eps = np.double(2 ** -15)
        self.N = N
        self.L = L
        self.a_int = np.array([1, -(1-eps)], dtype=np.double)
        b_comb = np.zeros(L*D + 1, dtype=np.double)
        b_comb[0] = 1
        b_comb[-1] = -((1 - eps) ** (L*D))
        self.b_comb = b_comb



    def forward(self, x):
        for n in range(self.N):
            x = lfilter(self.b_comb, self.a_int, x)
        return x

    def get_coeffs(self):
        '''
        Coefficients for single-stage in cascade
        :return: b, a
        '''
        return self.b_comb, self.a_int


class HSF():
    '''
    First-order high-shelf filter
    '''
    def __init__(self, G, fc, fs=2*np.pi, g=1.0):
        '''

        :param G: linear gain
        :param fc: cross-over frequency
        :param fs: sample rate
        :param g: scalar gain
        '''

        wc = 2 * np.pi * fc / fs
        sqrtG = np.sqrt(np.double(G))
        tanhwc2 = np.tan(np.double(wc) / 2)
        b0 = (sqrtG * tanhwc2 + G) / (sqrtG * tanhwc2 + 1)
        b1 = (sqrtG * tanhwc2 - G) / (sqrtG * tanhwc2 + 1)
        a1 = (sqrtG * tanhwc2 - 1) / (sqrtG * tanhwc2 + 1)
        self.b = g * np.stack([b0, b1])
        self.a = np.stack([1.0, a1])
        self.g = g

    def forward(self, x):
        return lfilter(self.b, self.a, x)

    def get_coeffs(self):
        return self.b, self.a
