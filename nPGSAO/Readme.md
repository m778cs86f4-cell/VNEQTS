# Edge Server Placement Experiments

This project evaluates edge-server (ES) placement on the Shanghai Telecom and Milan base-station datasets. The Shanghai experiment compares Random, SGA, PSO,and nPGSAO. The Milan entry currently runs SGA, PSO, and nPGSAO.

## Requirements

- Python 3.9 or later
- NumPy
- pandas

Windows PowerShell setup:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install numpy pandas
```

## Datasets

- Shanghai: `bs_statistics_vne_acu.csv`
- Milan: `milan_bs_workload.csv`

The Shanghai loader accepts base-station ID, latitude, longitude, and either a
`lambda` column or a total-duration column. The Milan loader expects `bs_id`,
`latitude`, `longitude`, and `workload`.

## Run the Shanghai Experiment

The default experiment uses 71 ACUs, a population of 30, 500 iterations,  and equal fitness weights.

```powershell
.\venv\Scripts\python.exe main.py
```

## Run the Milan Experiment

The Milan experiment uses 71 ACUs, a population of 30, 500 iterations,and equal fitness weights.

```powershell
.\venv\Scripts\python.exe main_milan.py --data .\milan_bs_workload.csv
```

## Model Rules

- A deployment contains exactly 71 ACUs.
- At most three ACUs may be deployed at one candidate site.
- Base-station demands are processed from largest to smallest.
- A demand is assigned only when its nearest active ES is in range and has
  sufficient residual capacity; otherwise it immediately falls back to cloud.
- Fitness uses equal weights for site coverage, traffic coverage, load balance,
  and energy efficiency.

## Output

The terminal prints the final values for:

- Fitness
- Site coverage
- Traffic coverage
- Load balance
- Energy efficiency
- Average utilization

Shanghai results are written to a unique directory under `shanghai_results/`.
Milan results are written to `milan_results/` by default. CSV outputs contain
deployment details, base-station allocations, convergence history, and summary metrics.

## Main Files

- `main.py`: Shanghai experiment entry point
- `main_milan.py`: Milan experiment entry point
- `fitness.py`: shared fitness, allocation, and cloud-fallback model
- `init_sys.py`: Shanghai data loading and distance calculation
- `evolution.py`: SGA implementation
- `swarm.py`: PSO implementation
- `niching.py`: nPGSAO and its crossover utilities

