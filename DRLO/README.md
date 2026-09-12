# DRLO Experiments

This project runs the DRLO baseline for two datasets:

- Shanghai Telecom dataset: `bs_statistics_all_12files.csv`
- Milan dataset: `milan_bs_workload.csv`

DRLO is trained separately on each dataset. 

## Run Shanghai Telecom Experiment

`train.py` reads the settings from `parameter.json`.

Run:

```powershell
.\venv\Scripts\python.exe train.py
```

For the Shanghai experiment, make sure `parameter.json` uses:

```json
"data_path": "bs_statistics_all_12files.csv"
```

## Run Milan Experiment

Run:

```powershell
.\venv\Scripts\python.exe run_milan_drlo.py
```

This script runs DRLO on the Milan dataset with Milan-specific settings.

## Main Output Metrics

The terminal and CSV files report:

```text
Fitness
Site coverage
Traffic coverage
Load balance
Energy efficiency
Total energy
Cloud fallback BSs
Active edge sites
Average utilization
```

## Output Files

Typical output files include:

```text
convergence_DRLO*.csv
acu_deployment_DRLO*.csv
bs_es_allocation_DRLO*.csv
weight_profile_summary_DRLO*.csv
```

## Notes

- Use `train.py` for the Shanghai main experiment and weight experiments.
- Use `run_drlo_shanghai_acu_sweep.py` for Shanghai ACU-number tests.
- Use `run_milan_drlo.py` for the Milan dataset.
