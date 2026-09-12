# VNEQTS Experiment

This repository contains a self-contained C++ implementation of a tuned VNEQTS-based edge server placement experiment. The program distributes a fixed budget of atomic compute units (ACUs) among candidate base-station sites and evaluates each deployment using site coverage, traffic coverage, load balance, and energy efficiency.

## Main Features

- Edge server placement based on a fixed ACU budget.
- Fitness evaluation using four objectives:
  - site coverage,
  - traffic coverage,
  - load balance,
  - energy efficiency.
- Von Neumann entropy based population diversity measurement.
- Adaptive rotation angle, mutation probability, and tabu list length.
- Quantum tabu local search.
- Spectral-guided escape mechanism for low-entropy stagnation.
- CSV outputs for convergence, deployment, allocation, and final summary.

## Requirements

- C++17 compatible compiler
- No external libraries are required

Example compiler:

```bash
g++ -std=c++17 -O2 vneqts.cpp -o vneqts
```

## Input CSV Format

The input CSV file should contain the following information for each base station:

- base station ID,
- latitude,
- longitude,
- workload or aggregated session duration.

Accepted column names include:

- ID: `bs_id`, `base_station_id`, or `id`
- Latitude: `latitude` or `lat`
- Longitude: `longitude`, `lon`, or `lng`
- Duration: `total_duration`, `duration`, or `session_duration`
- Workload: `lambda`, `lambda_k`, or `workload`

## Usage

```bash
./vneqts <workload.csv> <seed> <generations> <output_prefix>
```

Example:

```bash
./vneqts bs_statistics_all_12files.csv 1000 500 result_vneqts
```

Arguments:

- `<workload.csv>`: input base-station workload file
- `<seed>`: random seed 
- `<generations>`: number of evolutionary generations
- `<output_prefix>`: prefix used for output files

## Important Parameters

The main experimental parameters are defined near the beginning of the source file:

| Parameter | Value | Description |
|---|---:|---|
| `ACU_BUDGET` | 71 | Total number of ACUs to deploy |
| `MAX_ACUS_PER_SITE` | 3 | Maximum ACUs allowed at one site |
| `POPULATION_SIZE` | 30 | Population size |
| `DEFAULT_MAX_GENERATIONS` | 500 | Default number of generations |
| `NUM_MEASUREMENTS` | 50 | Number of QTS measurements |

## Output Files

For an output prefix such as `result_vneqts`, the program generates:

| Output file | Description |
|---|---|
| `result_vneqts_convergence.csv` | Convergence history and adaptive parameter values |
| `result_vneqts_deployment.csv` | Active edge sites, ACU counts, capacity, workload, and utilization |
| `result_vneqts_allocation.csv` | Base-station-to-edge-server assignment results |

## Fitness Function

The final fitness is calculated as a weighted sum of four normalized objectives:

```text
fitness = W_SITE * site_coverage
        + W_TRAFFIC * traffic_coverage
        + W_BALANCE * load_balance
        + W_ENERGY * energy_efficiency
```

In this version, the four weights are all set to `0.25`.

## Notes

- The escape mechanism is triggered only when the search is still within the active escape window, the population entropy is below the low-entropy threshold, and the global best solution has stagnated for enough generations.
