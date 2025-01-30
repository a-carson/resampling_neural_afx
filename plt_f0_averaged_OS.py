import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from glob import glob
import matplotlib.colors as mcolors

f = 'results    '
ylims = {
    'ESR': [-200, 40],
    'MESR': [-200, 40],
    'ASR': [-200, 40],
    'NMR': [-160, 80]
}

ax_dict = {
    'ESR': (0, 0),
    'MESR': (0, 1),
    'ASR': (1, 0),
    'NMR': (1, 1)
}

metrics = ['ESR', 'MESR', 'ASR', 'NMR']

exp_dir = 'results/combined_oversampling_20250123_110800'
custom_order = ['M=2_FFT', 'M=2_C-HB-IIR', 'M=2_C-HB-FIR',
                'M=4_FFT', 'M=4_C-HB-IIR', 'M=4_C-HB-FIR',
                'M=8_FFT', 'M=8_C-HB-IIR', 'M=8_C-HB-FIR', 'base']

model_dirs = sorted(os.listdir(exp_dir))
model_dirs = list(filter(lambda x: not '.py' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.csv' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.pdf' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.DS_Store' in x, model_dirs))

#model_dirs = list(filter(lambda x: not 'Ds-1' in x, model_dirs))


mean_dfs = {
    'ESR': pd.DataFrame(index=model_dirs),
    'MESR': pd.DataFrame(index=model_dirs),
    'ASR': pd.DataFrame(index=model_dirs),
    'NMR': pd.DataFrame(index=model_dirs)
}


for m, model_name in enumerate(model_dirs):

    csv_files = glob(f'{exp_dir}/{model_name}/*.csv')

    # Filter and sort
    csv_files = [str for str in csv_files if any(sub in str for sub in custom_order)]
    order_dict = {name: index for index, name in enumerate(custom_order)}
    csv_files = sorted(
        csv_files,
        key=lambda path: order_dict.get(path.split('/')[-1].split('.')[0], float('inf'))
    )
    data_col_start = 0
    for n, file in enumerate(csv_files):
        df = pd.read_csv(file, index_col=0)
        x_data = np.stack([float(x) for x in df.columns.values[data_col_start:]])
        method = os.path.split(file)[-1].split('.csv')[0]
        for idx, row in df.iterrows():
            if idx in metrics:
                y_data = row.values[data_col_start:]
                if idx == 'ASR':
                    mean_dfs[idx].at[model_name, method] = y_data[-1]
                elif idx == 'NMR':
                    mean_dfs[idx].at[model_name, method] = y_data[-1]
                else:
                    mean_dfs[idx].at[model_name, method] = np.mean(y_data)


# plot violins -- means first
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[6, 5.5])
for metric, df in mean_dfs.items():
    vplot = ax[ax_dict[metric]].violinplot(df.values,
                                           showmeans=True,
                                           showmedians=False,
                                           showextrema=False,
                                           widths=0.6)

    ax[ax_dict[metric]].vlines(3*np.arange(1, 4) + 0.5, ymin=ylims[metric][0], ymax=ylims[metric][1],
                               colors='k', alpha=0.1)
    ax[ax_dict[metric]].set_xticks([])
    ax[ax_dict[metric]].set_xlim(0.25, len(df.columns) + 0.75)
    ax[ax_dict[metric]].grid(alpha=0.25)
    for patch, color, method in zip(vplot['bodies'], list(mcolors.TABLEAU_COLORS), custom_order):
        if method == 'base':
            color = 'k'
        elif 'FFT' in method:
            color = 'tab:blue'
        elif 'IIR' in method:
            color = 'tab:orange'
        elif 'FIR' in method:
            color = 'tab:green'
        patch.set_facecolor(color)
        patch.set_edgecolor('none')
        patch.set_alpha(0.6)

    y = df.values
    x_ticks = np.arange(1, len(df.columns) + 1)
    for n, method in zip(range(y.shape[-1]), df.columns):
        if method == 'base':
            color = 'k'
        elif 'FFT' in method:
            color = 'tab:blue'
        elif 'IIR' in method:
            color = 'tab:orange'
        elif 'FIR' in method:
            color = 'tab:green'
        ax[ax_dict[metric]].scatter(np.ones_like(y[:, n]) + n - 0.0025, y[:, n],
                                    color=color,
                                    s=1.0,
                                    edgecolor='none')

    for partname in ('cmeans',):
        vp = vplot[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    ax[ax_dict[metric]].set_yticks(np.arange(ylims[metric][0], ylims[metric][1]+40, 40))
    ax[ax_dict[metric]].set_ylabel(metric + ' [dB]')
    ax[ax_dict[metric]].set_ylim(ylims[metric])

    props = dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.0)
    brace_height = 0.105
    brace_alpha = 0.25
    label_height = 0.055
    ax[ax_dict[metric]].text(0.062, brace_height, r'$\{$', transform=ax[ax_dict[metric]].transAxes, fontsize=30,
            verticalalignment='top', horizontalalignment='center', bbox=props, alpha=brace_alpha,
                             rotation=90,
                             rotation_mode='anchor')
    ax[ax_dict[metric]].text(0.15, label_height, '$M=2$', transform=ax[ax_dict[metric]].transAxes, fontsize=8, color='tab:grey',
            verticalalignment='top', horizontalalignment='center', bbox=props)

    ax[ax_dict[metric]].text(0.35, brace_height, r'$\{$', transform=ax[ax_dict[metric]].transAxes, fontsize=30,
            verticalalignment='top', horizontalalignment='center', bbox=props, alpha=brace_alpha,
                             rotation=90,
                             rotation_mode='anchor')
    ax[ax_dict[metric]].text(0.44, label_height, '$M=4$', transform=ax[ax_dict[metric]].transAxes, fontsize=8, color='tab:grey',
            verticalalignment='top', horizontalalignment='center', bbox=props)

    ax[ax_dict[metric]].text(0.64, brace_height, r'$\{$', transform=ax[ax_dict[metric]].transAxes, fontsize=30,
            verticalalignment='top', horizontalalignment='center', bbox=props, alpha=brace_alpha,
                             rotation=90,
                             rotation_mode='anchor')
    ax[ax_dict[metric]].text(0.73, label_height, '$M=8$', transform=ax[ax_dict[metric]].transAxes, fontsize=8, color='tab:grey',

            verticalalignment='top', horizontalalignment='center', bbox=props)
ax[ax_dict['NMR']].hlines(-10, xmin=0.25, xmax=len(df.columns) + 0.75, colors='k',
           linestyles='--',
           linewidth=0.5)


legend_labels = ['FFT', 'C-HB-IIR', 'C-HB-FIR', 'M=1']
legend_colors = ['tab:blue', 'tab:orange', 'tab:green', 'k']
legend = plt.legend(legend_labels, bbox_to_anchor=(-0.27, -0.25), loc='lower center',
           ncol=4, prop={'size': 8.75})

for handle, color in zip(legend.legend_handles, legend_colors):
    handle.set_facecolor(color)
    handle.set_edgecolor('none')
    handle.set_alpha(0.6)

fig.subplots_adjust(hspace=0.1, wspace=0.4)
plt.savefig(f'./{exp_dir}/violin_metrics_os.pdf', bbox_inches='tight')
plt.show()

