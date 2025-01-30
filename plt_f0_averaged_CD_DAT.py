import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from glob import glob
import matplotlib.colors as mcolors

ylims = {
    'ESR': [-200, 40],
    'MESR': [-200, 40],
    'ASR': [-200, 40],
    'NMR': [-200, 40]
}

ax_dict = {
    'ESR': (0, 0),
    'MESR': (0, 1),
    'ASR': (1, 0),
    'NMR': (1, 1)
}

metrics = ['ESR', 'ASR', 'MESR', 'NMR']

L = 160
if L == 160:
    exp_dir = 'results/L=160_M=147_20250122_153758'
    custom_order = ['NB-Kaiser', 'NB-Remez', 'HB-IIR+WB-Kaiser', 'HB-IIR+WB-Remez', 'LIDL', 'CIDL', 'naive', 'base']
    relace_idl_with_edl = False
    M = 147
elif L == 147:
    exp_dir = 'results/L=147_M=160_20250122_183322'
    custom_order = ['NB-Kaiser', 'NB-Remez', 'HB-IIR+WB-Kaiser', 'HB-IIR+WB-Remez', 'LIDL', 'CIDL', 'naive', 'base']
    relace_idl_with_edl = True
    M = 160




model_dirs = sorted(os.listdir(exp_dir))
model_dirs = list(filter(lambda x: not '.py' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.csv' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.pdf' in x, model_dirs))
model_dirs = list(filter(lambda x: not '.DS_Store' in x, model_dirs))

#model_dirs = list(filter(lambda x: not 'Ds-1' in x, model_dirs))


mean_dfs = {
    'ESR': pd.DataFrame(index=model_dirs),
    'ASR': pd.DataFrame(index=model_dirs),
    'MESR': pd.DataFrame(index=model_dirs),
    'NMR': pd.DataFrame(index=model_dirs)
}

dfs_4186 = {
    'ESR': pd.DataFrame(index=model_dirs),
    'ASR': pd.DataFrame(index=model_dirs),
    'MESR': pd.DataFrame(index=model_dirs),
    'NMR': pd.DataFrame(index=model_dirs)
}

dfs_27 = {
    'ESR': pd.DataFrame(index=model_dirs),
    'ASR': pd.DataFrame(index=model_dirs),
    'MESR': pd.DataFrame(index=model_dirs),
    'NMR': pd.DataFrame(index=model_dirs)
}

mean_dfs_across_model = {
    'ASR': pd.DataFrame(index=custom_order),
    'ESR': pd.DataFrame(index=custom_order),
    'MESR': pd.DataFrame(index=custom_order),
    'NMR': pd.DataFrame(index=custom_order)
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
                mean_dfs[idx].at[model_name, method] = np.mean(y_data)
                dfs_4186[idx].at[model_name, method] = np.max(y_data)
                dfs_27[idx].at[model_name, method] = y_data[0]

                # mean across model -- nasty method!
                for x, y in zip(x_data, y_data):
                    if f'{x}' in mean_dfs_across_model[idx].columns:
                        current = mean_dfs_across_model[idx].at[method, f'{x}']
                        if np.isnan(current):
                            current = 0
                    else:
                        current = 0
                    mean_dfs_across_model[idx].at[method, f'{x}'] = current + y / len(model_dirs)


# plot violins -- means first
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[6, 5])
for metric, df in mean_dfs.items():
    vplot = ax[ax_dict[metric]].violinplot(df.values,
                                           showmeans=True,
                                           showmedians=False,
                                           showextrema=False,
                                           widths=0.75)
    ax[ax_dict[metric]].set_xticks([],
               labels=[],
               rotation=90)
    ax[ax_dict[metric]].set_yticks(np.arange(ylims[metric][0], ylims[metric][1] + 40, 40))

    ax[ax_dict[metric]].set_xlim(0.25, len(df.columns) + 0.75)
    ax[ax_dict[metric]].grid(alpha=0.25)
    for patch, color, method in zip(vplot['bodies'], list(mcolors.TABLEAU_COLORS), custom_order):
        if method == 'base':
            color = 'k'
        elif method == 'naive':
            color = 'tab:grey'
        patch.set_facecolor(color)
        patch.set_edgecolor('none')
        patch.set_alpha(0.6)

    y = df.values
    x_ticks = np.arange(1, len(df.columns) + 1)
    for n in range(y.shape[-1]):
        if n == y.shape[-1] - 1:
            color = 'k'
        elif n == y.shape[-1] - 2:
            color = 'tab:grey'
        else:
            color = None
        ax[ax_dict[metric]].scatter(np.ones_like(y[:, n]) + n, y[:, n], color=color, s=1.0)

    for partname in ('cmeans',):
        vp = vplot[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    #ax[ax_dict[metric]].set_title(metric)
    ax[ax_dict[metric]].set_ylabel(metric + ' [dB]')
    ax[ax_dict[metric]].set_ylim(ylims[metric])


if relace_idl_with_edl:
    custom_order = [c.replace('IDL', 'EDL') for c in custom_order]

custom_order = [c.replace('SS', 'NB') for c in custom_order]
custom_order = [c.replace('MS', 'HB-IIR + WB') for c in custom_order]


legend_labels = custom_order
legend_labels[-1] = 'L=M=1'
plt.legend(custom_order, bbox_to_anchor=(-0.27, -0.3), loc='lower center',
           ncol=4,  prop={'size': 9})

fig.subplots_adjust(hspace=0.1, wspace=0.4)
plt.savefig(f'./{exp_dir}/violin_metrics_L={L}_M={M}.pdf', bbox_inches='tight')
plt.show()


# mean across models
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[6, 6])

for metric, df in mean_dfs_across_model.items():

    for method, row in df.iterrows():
        x_data = np.stack([float(x) for x in df.columns.values])
        y_data = row.values
        if 'base' in method:
            baseline = ax[ax_dict[metric]].semilogx(x_data, y_data, label=None, linestyle='--', color='k', linewidth=0.75)
        else:
            ax[ax_dict[metric]].semilogx(x_data, y_data, label=method, color='tab:grey' if 'naive' in method else None, linewidth=0.75)

for metric, ax_idx in ax_dict.items():
    ax[ax_idx].set_ylabel(metric + ' [dB]')
    ax[ax_idx].set_xlabel('f0 [Hz]')
    ax[ax_idx].grid(True, which='major', alpha=0.5)
    ax[ax_idx].grid(True, which='minor', alpha=0.1)
    ax[ax_idx].set_xlim([np.min(x_data), np.max(x_data)])
    ax[ax_idx].set_ylim(ylims[metric])

ax[0, 0].legend(loc='upper right', ncol=1, prop={'size': 8.5})
ax[1, 0].legend([baseline[0]], ['Original (L/M = 1)'], prop={'size': 8.5})

fig.subplots_adjust(hspace=0.3, wspace=0.4)
plt.savefig(f'./{exp_dir}/model_averaged_metrics.pdf', bbox_inches='tight')
#plt.suptitle(model_name)
plt.show()

print(mean_dfs)


interps = ['FFT', 'HB-IIR', 'HB-FIR', 'Linterp']
decims = ['FFT', 'HB-IIR', 'HB-FIR', 'CIC']
table = pd.DataFrame(index=interps, columns=decims)
df = dfs_4186['NMR']

df = df.filter(like='MesaMiniRec', axis=0)

for idx, row in table.iterrows():
    for col in table.columns:
        query = idx + '_' + col
        if query in df.columns:
            x = np.round(df[query].values[0], 1)
        else:
            x = 0
        table.at[idx, col] = x

print(table)