# ACO Edge Server Deployment

This folder contains Ant Colony Optimization (ACO) baselines for edge server
deployment on two datasets:

- Shanghai Telecom base-station dataset
- Milan base-station workload dataset

The algorithms search for an ACU deployment vector. Each candidate site may
receive several ACUs, and the objective combines site coverage, traffic
coverage, load balance, and energy efficiency.

## Requirements

- GNU Octave or MATLAB
- The input CSV files in the same folder as the scripts

## Shanghai Telecom ACO

Main script:

- `aco.m`

Required helper files:

- `loadBSModelVNEQTS.m`
- `resolveWeightProfileVNEQTS.m`
- `constructAcuSolutionACO.m`
- `evaluateDeploymentVNEQTS.m`
- `computeServiceMappingVNEQTS.m`
- `haversineMetersMatrix.m`

Input file:

- `bs_statistics_all_12files.csv`

Run command:

```bash
octave -qf --eval "run_id=0; weight_profile='A'; MaxIt=500; nAnt=50; aco;"
```

Notes:

- If no seed is provided, the script generates a random seed automatically.
- BSs that cannot be assigned to a feasible edge site fall back to the cloud.

Weight profiles:

- `A`: equal weights `[0.25, 0.25, 0.25, 0.25]`
- `B`: energy-prioritized weights
- `C`: coverage-prioritized weights

## Milan ACO

Main script:

- `aco_milan.m`

Input file:

- `milan_bs_workload.csv`

Run command:

```bash
octave -qf --eval "aco_milan('milan_bs_workload.csv', [], 500, 30, 71, 1500);"
```

Notes:

- If the seed argument is empty (`[]`), the script generates a random seed
  automatically.
- The Milan script is self-contained and includes its own data loading,
  distance calculation, ACO construction, fitness evaluation, and CSV output.

## Terminal Metrics

The final terminal output reports the main performance metrics:

- Fitness
- Site coverage
- Traffic coverage
- Load balance
- Energy efficiency
- Total energy
- Cloud fallback BSs
- Active edge sites
- Average utilization

## Output Files

Shanghai ACO writes files such as:

- `convergence_ACO_<profile>_run<id>.csv`
- `ACO_acu_deployment_<profile>_run<id>.csv`
- `ACO_bs_es_allocation_<profile>_run<id>.csv`
- `single_run_weight_profiles_ACO_VNEQTS.csv`

Milan ACO writes:

- `convergence_ACO_Milan.csv`
- `acu_deployment_ACO_Milan.csv`
- `bs_es_allocation_ACO_Milan.csv`

