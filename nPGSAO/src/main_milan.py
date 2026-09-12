
from argparse import ArgumentParser
from pathlib import Path
from time import time
import csv

import numpy as np
import pandas as pd

from fitness import fitness
from evolution import GA
from niching import nichingswarm
from swarm import PSO


DEFAULT_DATA_FILE = Path(
    r"D:\gec\gongxiang\AHUT\C++_code\BMCP_instance-master\ES"
    r"\Milan_VNE\milan_bs_workload.csv"
)
NUM_SERVERS =71
MAX_ACUS_PER_SITE = 3
ACU_CAPACITY = 11000.0
SERVICE_DISTANCE_KM = 1.5
POPULATION = 30
DEFAULT_ITERATIONS = 500
DEFAULT_SEED = 20260826
EQUAL_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def load_milan(csv_path):
    df = pd.read_csv(csv_path)
    required = {"bs_id", "latitude", "longitude", "workload"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Milan CSV is missing columns: {sorted(missing)}")

    workloads = df["workload"].to_numpy(dtype=float)
    if not np.all(np.isfinite(workloads)) or np.any(workloads < 0):
        raise ValueError("workload must contain finite, non-negative values")
    workloads = np.maximum(workloads, 1e-6)

    lat = np.radians(df["latitude"].to_numpy(dtype=float))
    lon = np.radians(df["longitude"].to_numpy(dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :])
         * np.sin(dlon / 2.0) ** 2)
    distances = 6371.0 * 2.0 * np.arctan2(
        np.sqrt(np.clip(a, 0.0, 1.0)),
        np.sqrt(np.clip(1.0 - a, 0.0, 1.0)),
    )

    capacities = np.full(len(df), ACU_CAPACITY, dtype=float)
    return df, workloads, capacities, distances


def history_to_f(history):
    return [1.0 - 1.0 / float(value) if value > 0 else 0.0
            for value in history]


def save_results(output_dir, algorithm_name, df, workloads, evaluator,
                 solution, history, seed, iterations, elapsed):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = evaluator._compute_metrics(solution)
    metrics = evaluator.unified_evaluate(solution)
    counts = raw["counts"]
    sigma = raw["sigma"]

    deployment = pd.DataFrame({
        "BS_ID": df.loc[counts > 0, "bs_id"].to_numpy(),
        "Latitude": df.loc[counts > 0, "latitude"].to_numpy(),
        "Longitude": df.loc[counts > 0, "longitude"].to_numpy(),
        "ACU_Count": counts[counts > 0],
        "Capacity": counts[counts > 0] * ACU_CAPACITY,
        "ES_Load": raw["site_loads"][counts > 0],
        "ES_Utilization_pct": (
            raw["site_loads"][counts > 0]
            / (counts[counts > 0] * ACU_CAPACITY)
        ) * 100.0,
    })
    deployment.to_csv(
        output_dir / f"{algorithm_name}_milan_deployment.csv",
        index=False, encoding="utf-8-sig",
    )

    serving_ids = np.full(len(df), "CLOUD", dtype=object)
    served = sigma >= 0
    serving_ids[served] = df["bs_id"].to_numpy()[sigma[served]]
    allocation = pd.DataFrame({
        "BS_ID": df["bs_id"],
        "Latitude": df["latitude"],
        "Longitude": df["longitude"],
        "Workload": workloads,
        "Local_ACU_Count": counts,
        "Serving_ES_ID": serving_ids,
        "Dist_to_ES_m": np.where(served, raw["nearest_dist"] * 1000.0, -1.0),
    })
    allocation.to_csv(
        output_dir / f"{algorithm_name}_milan_allocation.csv",
        index=False, encoding="utf-8-sig",
    )

    with open(output_dir / f"{algorithm_name}_milan_convergence.csv",
              "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Iteration", "F"])
        writer.writerows(enumerate(history_to_f(history)))

    summary_file = output_dir / "milan_equal_weight_results.csv"
    write_header = not summary_file.exists() or summary_file.stat().st_size == 0
    row = {
        "Algorithm": algorithm_name,
        "Seed": seed,
        "Iterations": iterations,
        "Population": POPULATION,
        "ACU_Budget": NUM_SERVERS,
        "ACU_Capacity": ACU_CAPACITY,
        "Max_ACUs_Per_Site": MAX_ACUS_PER_SITE,
        "Service_Distance_m": SERVICE_DISTANCE_KM * 1000.0,
        "W_site": evaluator.weights[0],
        "W_traf": evaluator.weights[1],
        "W_balance": evaluator.weights[2],
        "W_energy": evaluator.weights[3],
        **metrics,
        "Elapsed_seconds": round(elapsed, 3),
    }
    pd.DataFrame([row]).to_csv(
        summary_file, mode="a", header=write_header, index=False,
        encoding="utf-8-sig",
    )
    return metrics


def parse_args():
    parser = ArgumentParser(description="Run SGA, PSO, and nPGSAO on Milan")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--algorithm",
                        choices=["all", "SGA", "PSO", "nPGSAO"],
                        default="all")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("milan_results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be at least 1")

    df, workloads, capacities, distances = load_milan(args.data)

    evaluator = fitness(
        capacities, workloads, distances, weights=EQUAL_WEIGHTS,
        service_distance_km=SERVICE_DISTANCE_KM,
        acu_budget=NUM_SERVERS,
        exclude_zero_load_from_balance=True,
    )
    algorithms = [
        ("SGA", lambda: GA(
            evaluator, dim=NUM_SERVERS, high=len(df),
            ite=args.iterations, num_pop=POPULATION,
            prob_cross=0.8, prob_mu=0.1,
        ), "ga"),
        ("PSO", lambda: PSO(
            evaluator, dim=NUM_SERVERS, high=len(df),
            ite=args.iterations, num_pop=POPULATION,
            w_low=0.4, w_high=1.2, a1=2.0, a2=2.0,
        ), "run"),
        ("nPGSAO", lambda: nichingswarm(
            evaluator, dim=NUM_SERVERS, high=len(df),
            ite=args.iterations, num_pop=POPULATION,
            w_low=0.4, w_high=1.2, a1=2.0, a2=2.0,
        ), "nPGSAO"),
    ]
    if args.algorithm != "all":
        algorithms = [item for item in algorithms
                      if item[0] == args.algorithm]
    for algorithm_name, factory, method_name in algorithms:
        np.random.seed(args.seed)
        algorithm = factory()
        started = time()
        getattr(algorithm, method_name)()
        elapsed = time() - started
        solution = algorithm.final_best.copy()
        if (algorithm_name == "nPGSAO"
                and not evaluator.is_feasible_solution(solution)):
            raise RuntimeError(
                "nPGSAO: the final population contains no feasible best solution"
            )
        metrics = save_results(
            args.output_dir, algorithm_name, df, workloads, evaluator,
            solution, algorithm.history, args.seed, args.iterations, elapsed,
        )
        print(f"\n{algorithm_name}")
        print(f"Fitness: {metrics['F']:.6f}")
        print(f"Site coverage: {metrics['phi_site_pct']:.4f}%")
        print(f"Traffic coverage: {metrics['phi_traf_pct']:.4f}%")
        print(f"Load balance: {metrics['balance']:.6f}")
        print(f"Energy efficiency: {metrics['energy_eff']:.6f}")
        print(f"Average utilization: {metrics['avg_utilization_pct']:.4f}%")


if __name__ == "__main__":
    main()
