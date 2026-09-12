from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import uuid

import numpy as np
import pandas as pd

from evolution import GA
from fitness import fitness
from init_sys import init_sys
from niching import nichingswarm
from swarm import PSO


MAX_ITE = 500
P = 30
ACU_BUDGET = 71
SERVICE_DISTANCE_KM = 4.8
DATA_FILE = Path(__file__).resolve().parent / "bs_statistics_vne_acu.csv"
EQUAL_WEIGHTS = (0.25, 0.25, 0.25, 0.25)

def get_final_population_solution(alg_obj):
    return np.copy(alg_obj.pop[int(np.argmax(alg_obj.fits))])


def fitness_history_to_f(history_values):
    f_values = []
    for value in history_values:
        value = float(value)
        f_values.append(1.0 - 1.0 / value if value > 0 else 0.0)
    return f_values


def random_baseline(fit_ev, max_ite):
    solution = fit_ev.random_solution()
    f_value = fit_ev._compute_metrics(solution)["F"]
    history = [f_value] * (max_ite + 1)
    return solution, history


def save_and_report(algo_name, solution, fit_ev, req_rate, df,
                    output_dir, search_evaluations):
    metrics = fit_ev.unified_evaluate(solution)
    raw = fit_ev._compute_metrics(solution)
    counts = raw["counts"]
    sigma = raw["sigma"]
    site_loads = raw["site_loads"]
    nearest_dist = raw["nearest_dist"]

    bs_id_col = "bs_id" if "bs_id" in df.columns else df.columns[0]
    lat_col = "latitude" if "latitude" in df.columns else df.columns[1]
    lon_col = "longitude" if "longitude" in df.columns else df.columns[2]
    bs_ids = df[bs_id_col].values
    lats = df[lat_col].values
    lons = df[lon_col].values

    deployment_rows = []
    for site in np.flatnonzero(counts > 0):
        capacity = fit_ev.c0 * counts[site] * 2
        load = site_loads[site]
        if load > capacity + 1e-9:
            raise RuntimeError(f"{algo_name}: ES capacity constraint violated")
        rho = load / capacity if capacity > 0 else 0.0
        deployment_rows.append({
            "BS_ID": bs_ids[site],
            "Latitude": lats[site],
            "Longitude": lons[site],
            "ACU_Count": int(counts[site]),
            "Capacity": round(capacity, 6),
            "ES_Lambda": round(load, 6),
            "ES_Utilization_pct": round(rho * 100, 2),
        })

    deploy_file = output_dir / f"{algo_name}_shanghai_acu_deployment.csv"
    pd.DataFrame(deployment_rows).to_csv(deploy_file, index=False, encoding="utf-8-sig")

    allocation_rows = []
    for k in range(len(req_rate)):
        serving_site = int(sigma[k])
        allocation_rows.append({
            "BS_ID": bs_ids[k],
            "Latitude": lats[k],
            "Longitude": lons[k],
            "Lambda": round(float(req_rate[k]), 6),
            "ACU_Count": int(counts[k]),
            "Serving_ES_ID": bs_ids[serving_site] if serving_site >= 0 else "CLOUD",
            "Dist_to_ES_m": round(float(nearest_dist[k] * 1000), 1) if serving_site >= 0 else -1.0,
        })

    alloc_file = output_dir / f"{algo_name}_shanghai_bs_acu_allocation.csv"
    pd.DataFrame(allocation_rows).to_csv(alloc_file, index=False, encoding="utf-8-sig")

    print(f"\n{algo_name}")
    print(f"Fitness: {metrics['F']:.6f}")
    print(f"Site coverage: {metrics['phi_site_pct']:.4f}%")
    print(f"Traffic coverage: {metrics['phi_traf_pct']:.4f}%")
    print(f"Load balance: {metrics['balance']:.6f}")
    print(f"Energy efficiency: {metrics['energy_eff']:.6f}")
    print(f"Average utilization: {metrics['avg_utilization_pct']:.4f}%")

    row = {
        "Algorithm": algo_name, "Seed": "system_entropy",
        "Iterations": 0 if algo_name == "Random" else MAX_ITE,
        "Population": 1 if algo_name == "Random" else P,
        "SearchEvaluations": search_evaluations,
        "Dmax_m": SERVICE_DISTANCE_KM * 1000,
        "ACU_Budget": ACU_BUDGET,
        "W_site": fit_ev.weights[0], "W_traf": fit_ev.weights[1],
        "W_balance": fit_ev.weights[2], "W_energy": fit_ev.weights[3],
        **metrics,
    }
    result_file = output_dir / "shanghai_equal_weight_results.csv"
    write_header = not result_file.exists()
    with result_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return metrics


def main():
    np.random.seed(None)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = DATA_FILE.parent / "shanghai_results" / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "random_source": "system_entropy",
        "data_file": str(DATA_FILE),
        "data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "acu_budget": ACU_BUDGET, "dmax_m": SERVICE_DISTANCE_KM * 1000,
        "iterations": MAX_ITE, "population": P, "weights": EQUAL_WEIGHTS,
        "mapping": (
            "indivisible BS demands in descending-load order; use the nearest "
            "active site only when it is in range and has sufficient residual "
            "capacity; otherwise immediately use cloud"
        ),
        "capacity": "sum(lambda)/71", "selection_metric": "1/(1-F)",
        "capacity_constraint": "hard",
        "nPGSAO_constraint_handling": (
            "no repair; infeasible genotype receives zero fitness"
        ),
        "nPGSAO_crossover_source": "random current-population peer only",
        "reporting": "single Random deployment; final-generation population best for other algorithms",
        "sources": {
            name: hashlib.sha256((DATA_FILE.parent / name).read_bytes()).hexdigest()
            for name in ("main.py", "fitness.py", "init_sys.py",
                         "evolution.py", "swarm.py", "niching.py")
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    df_raw = pd.read_csv(DATA_FILE)
    num_bs, num_srv, _, lambdas, all_mus, dists, _, _ = init_sys(
        DATA_FILE, num_servers=ACU_BUDGET
    )
    fit_ev = fitness(
        all_mus, lambdas, dists, weights=EQUAL_WEIGHTS,
        service_distance_km=SERVICE_DISTANCE_KM, acu_budget=ACU_BUDGET,
    )
    labels = ["Iteration", "Random"]
    solution, random_history = random_baseline(fit_ev, MAX_ITE)
    save_and_report("Random", solution, fit_ev, lambdas, df_raw,
                    output_dir, 1)
    history = [list(range(MAX_ITE + 1)), random_history]
    classes = (GA, PSO, nichingswarm)
    algorithms = zip(
        ("SGA", "PSO", "nPGSAO"), classes, ("ga", "run", "nPGSAO")
    )
    for name, cls, method_name in algorithms:
        alg = cls(fit_ev, dim=num_srv, high=num_bs, ite=MAX_ITE, num_pop=P)
        getattr(alg, method_name)()
        solution = get_final_population_solution(alg)
        if name == "nPGSAO" and not fit_ev.is_feasible_solution(solution):
            raise RuntimeError(
                "nPGSAO: the final population contains no feasible best solution"
            )
        f_history = fitness_history_to_f(alg.history)
        actual_f = fit_ev._compute_metrics(solution)["F"]
        if len(f_history) != MAX_ITE + 1:
            raise RuntimeError(f"{name}: invalid convergence-history length")
        if not np.isclose(f_history[-1], actual_f, atol=1e-8, rtol=0):
            raise RuntimeError(f"{name}: reported solution and convergence disagree")
        save_and_report(name, solution, fit_ev, lambdas, df_raw, output_dir,
                        getattr(alg, "evaluation_count", ""))
        labels.append(name)
        history.append(f_history)
        with (output_dir / "converge_shanghai_equal_weight.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(labels)
            writer.writerows(zip(*history))


if __name__ == "__main__":
    main()
