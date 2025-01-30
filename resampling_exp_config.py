import numpy as np

from filters import Resampler, MultiStageResampler, Kaiser, Remez, \
    HalfbandIIR, HalfbandFIR, Lagrange, CIC, HSF

def config(id, pbr=0.5, sba=120, override_L=None):

    if id == 0:

        L = 160
        M = 147
        sr_base = 44100
        sr_input = 48000

        max_rate = sr_base * L

        halfband_args = {
            'pb_edge': 16e3,
            'ripple_sb': 118,
            'sr': 2 * sr_base
        }

        ss_kaiser_args = {
            'pb_edge': 11.5e3,
            'sb_edge': 28.1e3,
            'ripple_sb': sba,
            'sr': max_rate
        }

        ss_remez_args = {
            'pb_edge': 16e3,
            'sb_edge': 28.1e3,
            'ripple_pb': pbr,
            'ripple_sb': sba,
            'sr': max_rate
        }

        ms_kaiser_args = {
            'pb_edge': 0,
            'sb_edge': 2 * 44.1e3 - 28.1e3,
            'ripple_sb': sba,
            'sr': max_rate
        }

        ms_remez_args = {
            'pb_edge': 16e3,
            'sb_edge': 2 * 44.1e3 - 28.1e3,
            'ripple_sb': sba,
            'ripple_pb': pbr,
            'sr': max_rate
        }

        methods = [
            {
                'name': 'NB-Kaiser',
                'input': Resampler(L=M, M=L, filter_class=Kaiser, filter_args=ss_kaiser_args),
                'output': Resampler(L=L, M=M, filter_class=Kaiser, filter_args=ss_kaiser_args),
            },
            {
                'name': 'NB-Remez',
                'input': Resampler(L=M, M=L, filter_class=Remez, filter_args=ss_remez_args),
                'output': Resampler(L=L, M=M, filter_class=Remez, filter_args=ss_remez_args),
            },
            {
                'name': 'HB-IIR+WB-Kaiser',
                'input': MultiStageResampler([
                    Resampler(L=M, M=L // 2, filter_class=Kaiser, filter_args=ms_kaiser_args),
                    Resampler(L=1, M=2, filter_class=HalfbandIIR, filter_args=halfband_args),
                ]),
                'output': MultiStageResampler([
                    Resampler(L=2, M=1, filter_class=HalfbandIIR, filter_args=halfband_args),
                    Resampler(L=L // 2, M=M, filter_class=Kaiser, filter_args=ms_kaiser_args),
                ]),
            },
            {
                'name': 'HB-IIR+WB-Remez',
                'input': MultiStageResampler([
                    Resampler(L=M, M=L // 2, filter_class=Remez, filter_args=ms_remez_args),
                    Resampler(L=1, M=2, filter_class=HalfbandIIR, filter_args=halfband_args),
                ]),
                'output': MultiStageResampler([
                    Resampler(L=2, M=1, filter_class=HalfbandIIR, filter_args=halfband_args),
                    Resampler(L=L // 2, M=M, filter_class=Remez, filter_args=ms_remez_args),
                ]),
            },
            {
                'name': 'LIDL',
                'input': None,
                'output': None,
                'model_L': L,
                'model_M': M,
                'model_order': 1
            },
            {
                'name': 'CIDL',
                'input': None,
                'output': None,
                'model_L': L,
                'model_M': M,
                'model_order': 3
            },
            {
                'name': 'naive',
                'input': None,
                'output': None,
            }
        ]

    elif id == 1:

        L = 147
        M = 160
        sr_base = 48000
        sr_input = 44100

        max_rate = sr_base * L

        halfband_args = {
            'pb_edge': 16e3,
            'ripple_sb': 118,
            'sr': 2 * sr_base
        }

        ss_kaiser_args = {
            'pb_edge': 11.5e3,
            'sb_edge': 28.1e3,
            'ripple_sb': sba,
            'sr': max_rate
        }

        ss_remez_args = {
            'pb_edge': 16e3,
            'sb_edge': 28.1e3,
            'ripple_pb': pbr,
            'ripple_sb': sba,
            'sr': max_rate
        }

        ms_kaiser_args = {
            'pb_edge': 0,
            'sb_edge': 2 * 44.1e3 - 28.1e3,
            'ripple_sb': sba,
            'sr': max_rate
        }

        ms_remez_args = {
            'pb_edge': 16e3,
            'sb_edge': 2 * 44.1e3 - 28.1e3,
            'ripple_sb': sba,
            'ripple_pb': pbr,
            'sr': max_rate
        }

        methods = [
            {
                'name': 'NB-Kaiser',
                'input': Resampler(L=M, M=L, filter_class=Kaiser, filter_args=ss_kaiser_args),
                'output': Resampler(L=L, M=M, filter_class=Kaiser, filter_args=ss_kaiser_args),
            },
            {
                'name': 'NB-Remez',
                'input': Resampler(L=M, M=L, filter_class=Remez, filter_args=ss_remez_args),
                'output': Resampler(L=L, M=M, filter_class=Remez, filter_args=ss_remez_args),
            },
            {
                'name': 'HB-IIR+WB-Kaiser',
                'input': MultiStageResampler([
                    Resampler(L=2, M=1, filter_class=HalfbandIIR, filter_args=halfband_args),
                    Resampler(L=M // 2, M=L, filter_class=Kaiser, filter_args=ms_kaiser_args),
                ]),
                'output': MultiStageResampler([
                    Resampler(L=L, M=M // 2, filter_class=Kaiser, filter_args=ms_kaiser_args),
                    Resampler(L=1, M=2, filter_class=HalfbandIIR, filter_args=halfband_args),
                ]),

            },
            {
                'name': 'HB-IIR+WB-Remez',
                'input': MultiStageResampler([
                    Resampler(L=2, M=1, filter_class=HalfbandIIR, filter_args=halfband_args),
                    Resampler(L=M // 2, M=L, filter_class=Remez, filter_args=ms_remez_args),
                ]),
                'output': MultiStageResampler([
                    Resampler(L=L, M=M // 2, filter_class=Remez, filter_args=ms_remez_args),
                    Resampler(L=1, M=2, filter_class=HalfbandIIR, filter_args=halfband_args),
                ]),

            },
            {
                'name': 'LIDL',
                'input': None,
                'output': None,
                'model_L': L,
                'model_M': M,
                'model_order': 1
            },
            {
                'name': 'CIDL',
                'input': None,
                'output': None,
                'model_L': L,
                'model_M': M,
                'model_order': 3
            },
            {
                'name': 'naive',
                'input': None,
                'output': None,
            }
        ]

    elif id == 2 or id == 3:

        L = 8
        M = 1
        sr_base = 44100
        sr_input = 44100

        halfband_args = {
            'pb_edge': 16e3,
            'ripple_sb': sba-2,
            'sr': 2 * sr_base
        }

        num_cascades = int(np.log2(L))

        interps = {
            'FFT': Resampler(L=L, M=1, filter_class=None, filter_args=None, ideal=True),
            'C-HB-IIR': MultiStageResampler(
                    [Resampler(L=2, M=1, filter_class=HalfbandIIR, filter_args=halfband_args)] * num_cascades
                ),
            'C-HB-FIR': MultiStageResampler(
                    [Resampler(L=2, M=1, filter_class=HalfbandFIR, filter_args=halfband_args)] * num_cascades
                ),
            'Linterp':  Resampler(L=L, M=1, filter_class=Lagrange,
                                  filter_args={'L': L, 'N': 1},
                                  pre_filter=HSF(G=1.941, fc=16e3, fs=sr_base)),
        }

        decims = {
            'FFT': Resampler(L=1, M=L, filter_class=None, filter_args=None, ideal=True),
            'C-HB-IIR': MultiStageResampler(
                    [Resampler(L=1, M=2, filter_class=HalfbandIIR, filter_args=halfband_args)] * num_cascades
                ),
            'C-HB-FIR': MultiStageResampler(
                    [Resampler(L=1, M=2, filter_class=HalfbandFIR, filter_args=halfband_args)] * num_cascades
                ),
            'CIC': Resampler(L=1, M=L, filter_class=CIC, filter_args={'L': L},
                             post_filter=HSF(G=12.71, fc=16e3, fs=sr_base, g=3.817e-6))
        }

        methods = []
        if id == 2:
            for n in ['FFT', 'C-HB-IIR', 'C-HB-FIR']:
                methods.append({
                    'name': n,
                    'input': interps[n],
                    'output': decims[n],
                    'model_L': L,
                    'model_M': 1,
                })
            methods.append({
                'name': 'EQ-Linterp + CIC',
                'input': interps['Linterp'],
                'output': decims['CIC'],
                'model_L': L,
                'model_M': 1,
            })
        if id == 3:
            for i_name, i_filter in interps.items():
                for d_name, d_filter in decims.items():
                    methods.append({
                        'name': i_name + '_' + d_name,
                        'input': i_filter,
                        'output': d_filter,
                        'model_L': L,
                        'model_M': 1,
                    })

    elif id == 4:

        L = override_L
        M = 1
        sr_base = 44100
        sr_input = 44100

        halfband_args = {
            'pb_edge': 16e3,
            'ripple_sb': sba-2,
            'sr': 2 * sr_base
        }
        #
        # fir_args = {
        #     'pb_edge': 11.5e3,
        #     'sb_edge': 28.1e3,
        #     'ripple_sb': sba,
        #     'sr': 2 * sr_base
        # }

        num_cascades = int(np.log2(L))

        iir_interpolator = MultiStageResampler(
                    [Resampler(L=2, M=1, filter_class=HalfbandIIR, filter_args=halfband_args)] * num_cascades
                )
        iir_decimator = MultiStageResampler(
                    [Resampler(L=1, M=2, filter_class=HalfbandIIR, filter_args=halfband_args)] * num_cascades
                )

        fir_interpolator = MultiStageResampler(
                    [Resampler(L=2, M=1, filter_class=HalfbandFIR, filter_args=halfband_args)] * num_cascades
                )
        fir_decimator = MultiStageResampler(
                    [Resampler(L=1, M=2, filter_class=HalfbandFIR, filter_args=halfband_args)] * num_cascades
                )

        perfect_interpolator = Resampler(L=L, M=1, filter_class=None, filter_args=None, ideal=True)
        perfect_decimate = Resampler(L=1, M=L, filter_class=None, filter_args=None, ideal=True)

        methods = [
            {
                'name': 'FFT',
                'input': perfect_interpolator,
                'output': perfect_decimate,
                'model_L': L,
                'model_M': 1,
            },
            {
                'name': 'C-HB-IIR',
                'input': iir_interpolator,
                'output': iir_decimator,
                'model_L': L,
                'model_M': 1,
            },
            {
                'name': 'C-HB-FIR',
                'input': fir_interpolator,
                'output': fir_decimator,
                'model_L': L,
                'model_M': 1,
            },
        ]

    else:
        raise Exception('Error: invalid experiment ID')

    cfg = {
        'L': L,
        'M': M,
        'sr_input': sr_input,
        'sr_base': sr_base,
        'methods': methods
    }

    return cfg