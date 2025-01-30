import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from argparse import ArgumentParser


ax_dict = {
    'ESR': (0, 0),
    'MESR': (0, 1),
    'ASR': (1, 0),
    'NMR': (1, 1)
}
ylims = [-200, 40]

metrics = ['SNR', 'SNRH', 'SNRA', 'NMR']

parser = ArgumentParser()
parser.add_argument('--exp_no', type=int, default=0)
args = parser.parse_args()

if args.exp_no == 0:
    exp_dir = 'results/L=160_M=147_20250122_153758'
    custom_order = ['NB-Kaiser', 'NB-Remez', 'HB-IIR+WB-Kaiser', 'HB-IIR+WB-Remez', 'LIDL', 'CIDL', 'naive', 'base']
    relace_idl_with_edl = False
elif args.exp_no == 1:
    exp_dir = 'results/L=147_M=160_20250122_183322'
    custom_order = ['NB-Kaiser', 'NB-Remez', 'HB-IIR+WB-Kaiser', 'HB-IIR+WB-Remez', 'LIDL', 'CIDL', 'naive', 'base']
    relace_idl_with_edl = True
elif args.exp_no == 2:
    relace_idl_with_edl = False
    exp_dir = 'results/L=8_M=1_20250122_163755'
    custom_order = ['base', 'FFT', 'C-HB-IIR', 'C-HB-FIR', 'EQ-Linterp + CIC']
elif args.exp_no == 3:
    relace_idl_with_edl = False
    exp_dir = 'results/L=8_M=1_20250129_152118'
    custom_order = ['base', 'FFT', 'C-HB-IIR', 'C-HB-FIR', 'CIC']


model_dirs = sorted(os.listdir(exp_dir))
model_dirs = list(filter(lambda x: not '.py' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.csv' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.pdf' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.DS_Store' in x, model_dirs))

mean_dfs = {
    'ESR': pd.DataFrame(index=model_dirs),
    'MESR': pd.DataFrame(index=model_dirs),
    'ASR': pd.DataFrame(index=model_dirs),
    'NMR': pd.DataFrame(index=model_dirs)
}

for model_name in model_dirs:

    csv_files = glob(f'{exp_dir}/{model_name}/*.csv')

    # Filter and sort
    csv_files = [str for str in csv_files if any(sub in str for sub in custom_order)]
    order_dict = {name: index for index, name in enumerate(custom_order)}
    csv_files = sorted(
        csv_files,
        key=lambda path: order_dict.get(path.split('/')[-1].split('.')[0], float('inf'))
    )

    data_col_start = 0
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[6, 5])

    for file in csv_files:
        df = pd.read_csv(file, index_col=0)
        x_data = np.stack([float(x) for x in df.columns.values[data_col_start:]])
        method = os.path.split(file)[-1].split('.csv')[0]
        for idx, row in df.iterrows():
            if idx in ax_dict:
                y_data = row.values[data_col_start:]
                if 'base' in method:
                    baseline = ax[ax_dict[idx]].semilogx(x_data, y_data, label=None,
                                                         linewidth=0.75,
                                                         linestyle='--', color='k')
                else:
                    color = 'tab:grey' if 'naive' in method else None

                    m = method
                    if relace_idl_with_edl:
                        m = m.replace('IDL', 'EDL')
                    ax[ax_dict[idx]].semilogx(x_data, y_data, label=m, color=color, linewidth=0.75)

                # MEANS
                mean_dfs[idx].at[model_name, method] = np.mean(y_data)

    for metric, ax_idx in ax_dict.items():
        ax[ax_idx].set_ylabel(metric + ' [dB]')
        ax[ax_idx].grid(True, which='major', alpha=0.5)
        ax[ax_idx].grid(True, which='minor', alpha=0.1)
        ax[ax_idx].set_xlim([np.min(x_data), np.max(x_data)])
        ax[ax_idx].set_ylim(ylims)
        ax[ax_idx].set_yticks(np.arange(ylims[0], ylims[1]+40, 40))
        #ax[ax_idx].set_xticklabels([])

    ax[0, 0].set_xticklabels([])
    ax[0, 1].set_xticklabels([])
    ax[1, 0].set_xlabel('$f_0$ [Hz]')
    ax[1, 1].set_xlabel('$f_0$ [Hz]')
    ax[0, 0].legend(loc='lower right', ncol=1, prop={'size': 7})
    ax[1, 0].legend([baseline[0]], ['L=M=1'], prop={'size': 7})
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    plt.savefig(f'./{exp_dir}/{model_name}_metrics.pdf', bbox_inches='tight')
    plt.suptitle(model_name)
    plt.show()


# plot means
for metric, df in mean_dfs.items():
    df.to_csv(f'{exp_dir}/{metric}.csv')
print(mean_dfs)


interps = ['FFT', 'C-HB-IIR', 'C-HB-FIR', 'Linterp']
decims = ['FFT', 'C-HB-IIR', 'C-HB-FIR', 'CIC']
table = pd.DataFrame(index=interps, columns=decims)
for idx, row in table.iterrows():
    for col in table.columns:
        query = idx + '_' + col
        if query in df.columns:
            x = np.round(df[query].values[0], 3)
        else:
            x = 0
        table.at[idx, col] = x

print(table)

