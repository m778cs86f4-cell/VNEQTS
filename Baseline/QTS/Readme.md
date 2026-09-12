# Quantum-Inspired Tabu Search for Edge Server Placement

## Overview

This repository contains Quantum-Inspired Tabu Search (QTS) baselines for the Edge Server Placement Problem (ESPP) on two cellular-network datasets:

- Shanghai Telecom: `bs_statistics_all_12files.csv`
- Telecom Italia Milan: `milan_bs_workload.csv`

The QTS solution uses binary quantum-inspired variables. Each candidate site has three binary ACU slots, and the number of selected slots determines the number of ACUs deployed at that site. 

## Main files

| Dataset | Main program | Input data |
|---|---|---|
| Shanghai Telecom | `qts_espp_baseline_with_t0.py` | `bs_statistics_all_12files.csv` |
| Milan | `QTS_Milan.py` | `milan_bs_workload.csv` |

## Requirements

- Python 3.10 or later
- NumPy

Install the core dependency with:

```powershell
python -m pip install numpy
```

The core algorithms use only NumPy and the Python standard library. 

## Common QTS configuration

| Parameter | Default |
|---|---:|
| Iterations | 500 |
| Neighborhood measurements per iteration | 50 |
| Rotation angle | `0.01*pi` |
| Independent runs | 1 |
| Random seed | 20260823 |
| Total ACU budget | 71 |
| Maximum ACUs at one site | 3 |

## Evaluation metrics

- `Phi_site`: fraction of base stations assigned to an edge site.
- `Phi_traf`: fraction of workload assigned to edge sites.
- `B`: load-balance score derived from the coefficient of variation of utilization.
- `Ee`: normalized energy-efficiency score.
- `F`: weighted sum of `Phi_site`, `Phi_traf`, `B`, and `Ee`.
- `Mean util.`: mean utilization of active edge sites.

## Shanghai Telecom experiment

### Data format

The first four CSV columns must contain:

```text
bs_id, latitude, longitude, total_duration(s)
```

The program recomputes workload as:

```text
lambda_k = max(total_duration / 15,811,200, 1e-6)
```

### System model

- Base stations: 3,007 in the current dataset.
- Total ACU budget: 71.
- Maximum ACUs per site: 3

### Weight profiles

The weight order is `[Phi_site, Phi_traf, B, Ee]`.

| Profile | Weights | Purpose |
|---|---|---|
| A | `[0.25, 0.25, 0.25, 0.25]` | Equal weighted |
| B | `[0.15, 0.15, 0.20, 0.50]` | Energy prioritized |
| C | `[0.40, 0.40, 0.10, 0.10]` | Coverage prioritized |

### Standard single run

```powershell
python qts_espp_baseline_with_t0.py `
  --data bs_statistics_all_12files.csv `
  --iterations 500 `
  --neighbors 50 `
  --runs 1 `
  --seed 20260823 `
  --init-mode random `
  --weight-profile A `
  --output-dir qts_weight_profile_A_results
```

Run profiles B and C by replacing `A` and the output-directory name.

### Shanghai outputs

- `convergence_QTS.csv`: best-so-far convergence samples at `t=10,20,...,500` for a single run.
- `convergence_QTS_runNN.csv`: per-run convergence files when `--runs` is greater than one.
- `convergence_QTS_mean.csv`: mean and standard deviation across multiple runs.
- `qts_runs_summary.csv`: final metrics and run metadata.
- `acu_deployment_QTS.csv`: ACUs deployed at candidate sites.
- `bs_es_allocation_QTS.csv`: BS-to-ES assignment or cloud fallback.

## Milan experiment

### Data format

The Milan CSV must include recognizable columns for:

```text
bs_id, latitude, longitude, workload
```

The current file contains 2,500 grid-centroid base-station records. The accompanying `milan_bs_workload.metadata.json` records the preprocessing provenance and defines workload as mean Internet activity per nonblank ten-minute slot.

### System model

- Maximum service distance: 1,500 m.
- Total ACU budget: 71.
- Maximum ACUs per site: 3.
- Fixed ACU capacity: 11,000 workload units.
- Fitness weights: `[0.25, 0.25, 0.25, 0.25]`.

### Standard single run

```powershell
python QTS_Milan.py `
  --data milan_bs_workload.csv `
  --iterations 500 `
  --neighbors 50 `
  --runs 1 `
  --seed 20260823 `
  --verbose-every 0 `
  --output-dir qts_milan_results
```

### Milan outputs

- `convergence_QTS_Milan.csv`: single-run convergence history, including `t=0` and every iteration.
- `convergence_QTS_Milan_runNN.csv`: per-run convergence history for multiple runs.
- `convergence_QTS_Milan_mean.csv`: mean and standard deviation across runs.
- `qts_milan_runs_summary.csv`: final run statistics.
- `acu_deployment_QTS_Milan.csv`: ACU deployment.
- `bs_es_allocation_QTS_Milan.csv`: BS allocation and cloud fallback.
