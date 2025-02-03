# Resampling Filter Design for Multirate Neural Audio Effects Processing



#### Requirements

Create conda environment:
```
conda env create -f conda_env.yaml
conda activate resampling_neural_afx
```


#### Run experiments
Run the experiments described in the paper, in the order that they appear:
```angular2html
python3 resampling_exp.py --exp_no 0 --model_paths 'Proteus_Tone_Packs/Selection*.json' --log_results
python3 resampling_exp.py --exp_no 1 --model_paths 'AIDA-X/*.json' --log_results
python3 resampling_exp.py --exp_no 2 --model_paths Proteus_Tone_Packs/HighGain/MesaMiniRec_HighGain_DirectOut.json --log_results
python3 resampling_exp.py --exp_no 3 --model_paths Proteus_Tone_Packs/HighGain/MesaMiniRec_HighGain_DirectOut.json --log_results
python3 resampling_exp.py --exp_no 4 --model_paths 'Proteus_Tone_Packs/HighGain/*.json' --log_results
```

#### Generating plots
Generate figures in the paper (and more):

Filter magnitude responses:
```angular2html
python3 plt_filter_analysis_L=160_M=147.py
python3 plt_filter_analysis_M=8.py
```

Metrics vs sine tone frequency:
```angular2html
python3 plt_metrics_vs_f0.py --exp_no 0
python3 plt_metrics_vs_f0.py --exp_no 1
python3 plt_metrics_vs_f0.py --exp_no 2
python3 plt_metrics_vs_f0.py --exp_no 3
```

Violin plots:
```angular2html
python3 plt_f0_averaged_CD_DAT.py --exp_no 0
python3 plt_f0_averaged_CD_DAT.py --exp_no 1
python3 plt_f0_averaged_OS.py
```