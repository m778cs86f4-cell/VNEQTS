#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Parameters aligned with VNEQTS_Milan.cpp
# =============================================================================
DEFAULT_ITERATIONS = 500
DEFAULT_NEIGHBORS = 50
DEFAULT_THETA_PI = 0.01
DEFAULT_RUNS = 1
DEFAULT_SEED = 20260823

D_MAX = 1500.0
M_BUDGET = 71
N_MAX = 3
ACU_CAPACITY = 11_000.0
NUMERIC_EPS = 1e-10
B_EPS = 1e-9

W1 = 0.25
W2 = 0.25
W3 = 0.25
W4 = 0.25

EARTH_R = 6_371_000.0


# =============================================================================
# Data structures
# =============================================================================
@dataclass(frozen=True)
class BaseStationData:
    bs_id: np.ndarray           # dtype object/string
    lat: np.ndarray             # float64
    lon: np.ndarray             # float64
    workload: np.ndarray        # float64 Milan traffic proxy

    @property
    def n(self) -> int:
        return int(self.lat.size)


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    phi_site: float
    phi_traf: float
    balance: float
    energy_eff: float
    total_energy: float
    cloud_fallback: int


@dataclass
class QTSRunResult:
    run_id: int
    seed: int
    best_iteration: int
    best_bits: np.ndarray
    best_n: np.ndarray
    best_metrics: FitnessResult
    history: List[dict]
    logical_evaluations: int
    unique_evaluations: int
    runtime_sec: float


def read_base_stations(filename: Path) -> BaseStationData:
    """Read Milan IDs, grid-centroid coordinates, and precomputed workload."""
    ids: List[str] = []
    lats: List[float] = []
    lons: List[float] = []
    workloads: List[float] = []

    with filename.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {filename}")

        normalized = {name.strip().lower(): name for name in reader.fieldnames}

        def find_column(candidates: Sequence[str]) -> str:
            for candidate in candidates:
                source = normalized.get(candidate)
                if source is not None:
                    return source
            raise ValueError(
                f"{filename}: missing one of columns {list(candidates)}; "
                f"found {reader.fieldnames}"
            )

        id_col = find_column(("bs_id", "base_station_id", "id"))
        lat_col = find_column(("latitude", "lat"))
        lon_col = find_column(("longitude", "lon", "lng"))
        workload_col = find_column(("lambda", "lambda_k", "workload"))

        for line_no, row in enumerate(reader, start=2):
            if not row or all(not str(x).strip() for x in row.values() if x is not None):
                continue
            try:
                bs_id = str(row[id_col]).strip()
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                workload = float(row[workload_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{filename}: invalid Milan record at line {line_no}: {row}"
                ) from exc

            if (
                not bs_id
                or not math.isfinite(lat)
                or not math.isfinite(lon)
                or not math.isfinite(workload)
                or not -90.0 <= lat <= 90.0
                or not -180.0 <= lon <= 180.0
                or workload < 0.0
            ):
                raise ValueError(f"{filename}: out-of-range value at line {line_no}")
            workload = max(workload, 1e-6)

            ids.append(bs_id)
            lats.append(lat)
            lons.append(lon)
            workloads.append(workload)

    if not ids:
        raise ValueError(f"No base-station records found in {filename}")

    return BaseStationData(
        bs_id=np.asarray(ids, dtype=object),
        lat=np.asarray(lats, dtype=np.float64),
        lon=np.asarray(lons, dtype=np.float64),
        workload=np.asarray(workloads, dtype=np.float64),
    )


def build_distance_matrix(bs: BaseStationData, block_size: int = 256) -> np.ndarray:
    n = bs.n
    lat = np.radians(bs.lat)
    lon = np.radians(bs.lon)
    cos_lat = np.cos(lat)

    dist = np.empty((n, n), dtype=np.float32)

    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        lat_b = lat[start:end, None]
        lon_b = lon[start:end, None]

        dlat = lat[None, :] - lat_b
        dlon = lon[None, :] - lon_b
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat_b) * cos_lat[None, :] * np.sin(dlon / 2.0) ** 2
        )
        np.clip(a, 0.0, 1.0, out=a)
        d = EARTH_R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist[start:end, :] = d.astype(np.float32, copy=False)

    # Remove tiny numerical diagonal noise.
    np.fill_diagonal(dist, 0.0)
    return dist


class ESPPEvaluator:
    def __init__(self, bs: BaseStationData, dist_matrix: np.ndarray):
        self.bs = bs
        self.dist = dist_matrix
        self.total_workload = float(np.sum(bs.workload))
        if self.total_workload <= 0.0:
            raise ValueError("Total Milan workload must be positive.")
        self.acu_capacity = ACU_CAPACITY
        self.cache: Dict[bytes, FitnessResult] = {}
        self.logical_evaluations = 0
        self.unique_evaluations = 0

    def reset_counters_and_cache(self) -> None:
        self.cache.clear()
        self.logical_evaluations = 0
        self.unique_evaluations = 0

    @staticmethod
    def _key(n_vec: np.ndarray) -> bytes:
        return np.asarray(n_vec, dtype=np.uint8).tobytes()

    def service_mapping(self, n_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        active = np.flatnonzero(n_vec > 0)
        sigma = np.full(self.bs.n, -1, dtype=np.int32)
        min_dist = np.full(self.bs.n, np.inf, dtype=np.float32)

        if active.size == 0:
            return sigma, min_dist

        sub = self.dist[:, active]
        pos = np.argmin(sub, axis=1)
        mins = sub[np.arange(self.bs.n), pos]
        covered = mins <= D_MAX
        sigma[covered] = active[pos[covered]]
        min_dist[covered] = mins[covered]
        return sigma, min_dist

    def evaluate(self, n_vec: np.ndarray) -> FitnessResult:
        self.logical_evaluations += 1

        n_vec = np.asarray(n_vec, dtype=np.int8)
        key = self._key(n_vec)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        self.unique_evaluations += 1
        sigma, _ = self.service_mapping(n_vec)
        covered = sigma != -1

        covered_sites = int(np.count_nonzero(covered))
        covered_workload = float(np.sum(self.bs.workload[covered]))
        cloud_count = self.bs.n - covered_sites

        phi_site = covered_sites / self.bs.n
        phi_traf = covered_workload / self.total_workload

        loads = np.zeros(self.bs.n, dtype=np.float64)
        if covered_sites:
            np.add.at(loads, sigma[covered], self.bs.workload[covered])

        active = np.flatnonzero(n_vec > 0)
        if active.size:
            capacity = self.acu_capacity * n_vec[active].astype(np.float64)
            rho = np.divide(
                loads[active], capacity,
                out=np.zeros_like(capacity),
                where=capacity > 0,
            )
            rho = np.clip(rho, 0.0, 1.0)
            total_energy = float(
                np.sum(n_vec[active].astype(np.float64) * (1.0 + rho))
            )

            # Match VNEQTS_Milan.cpp: exclude active sites receiving no load
            # from the balance statistic, while retaining them in energy.
            positive_rho = rho[loads[active] > NUMERIC_EPS]
            if positive_rho.size:
                mean_rho = float(np.mean(positive_rho))
                std_rho = float(np.std(positive_rho, ddof=0))
                cv = std_rho / (mean_rho + B_EPS)
                balance = 1.0 - min(1.0, cv)
            else:
                balance = 0.0
        else:
            total_energy = 0.0
            balance = 0.0

        energy_eff = 1.0 - (total_energy - M_BUDGET) / M_BUDGET
        energy_eff = float(np.clip(energy_eff, 0.0, 1.0))

        fitness = (
            W1 * phi_site
            + W2 * phi_traf
            + W3 * balance
            + W4 * energy_eff
        )

        result = FitnessResult(
            fitness=float(fitness),
            phi_site=float(phi_site),
            phi_traf=float(phi_traf),
            balance=float(balance),
            energy_eff=float(energy_eff),
            total_energy=float(total_energy),
            cloud_fallback=int(cloud_count),
        )
        self.cache[key] = result
        return result

    def mean_utilization(self, n_vec: np.ndarray) -> float:
        sigma, _ = self.service_mapping(n_vec)
        covered = sigma != -1
        loads = np.zeros(self.bs.n, dtype=np.float64)
        if np.any(covered):
            np.add.at(loads, sigma[covered], self.bs.workload[covered])

        active = np.flatnonzero(n_vec > 0)
        if active.size == 0:
            return 0.0
        capacity = self.acu_capacity * n_vec[active].astype(np.float64)
        rho = np.divide(
            loads[active], capacity,
            out=np.zeros_like(capacity),
            where=capacity > 0,
        )
        rho = np.clip(rho, 0.0, 1.0)
        return float(np.mean(rho))

def decode_acu_slots(bits: np.ndarray, n_sites: int) -> np.ndarray:
    return bits.reshape(n_sites, N_MAX).sum(axis=1, dtype=np.int16).astype(np.int8)


def repair_exact_budget(bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.bool_).copy()
    total = int(np.count_nonzero(bits))

    if total > M_BUDGET:
        ones = np.flatnonzero(bits)
        drop = rng.choice(ones, size=total - M_BUDGET, replace=False)
        bits[drop] = False
    elif total < M_BUDGET:
        zeros = np.flatnonzero(~bits)
        add = rng.choice(zeros, size=M_BUDGET - total, replace=False)
        bits[add] = True

    return bits


def measure_solution(q_angles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p_one = np.sin(q_angles) ** 2
    return rng.random(q_angles.size) < p_one


def measure_neighborhood(
    q_angles: np.ndarray,
    neighborhood_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    p_one = np.sin(q_angles) ** 2
    return rng.random((neighborhood_size, q_angles.size)) < p_one[None, :]


def qts_move_gate_update(
    q_angles: np.ndarray,
    best_bits: np.ndarray,
    worst_bits: np.ndarray,
    theta: float,
) -> int:
    best_i = best_bits.astype(np.int8, copy=False)
    worst_i = worst_bits.astype(np.int8, copy=False)
    diff = best_i - worst_i       # +1: toward |1>, -1: toward |0>, 0: tabu
    changed = diff != 0

    if np.any(changed):
        alpha = np.cos(q_angles[changed])
        beta = np.sin(q_angles[changed])
        # +1 in quadrants I/III; -1 in II/IV.
        quadrant_sign = np.where(alpha * beta >= 0.0, 1.0, -1.0)
        q_angles[changed] += theta * diff[changed] * quadrant_sign

    return int(np.count_nonzero(~changed))


def metric_rank(m: FitnessResult) -> Tuple[float, float, float, float, float, float, int]:
    return (
        m.fitness,
        m.phi_site,
        m.phi_traf,
        m.balance,
        m.energy_eff,
        -m.total_energy,
        -m.cloud_fallback,
    )


def run_qts_baseline(
    evaluator: ESPPEvaluator,
    iterations: int,
    neighborhood_size: int,
    theta: float,
    seed: int,
    run_id: int = 1,
    verbose_every: int = 50,
) -> QTSRunResult:
    rng = np.random.default_rng(seed)
    n_sites = evaluator.bs.n
    n_qubits = n_sites * N_MAX

    p0 = M_BUDGET / n_qubits
    q0 = math.asin(math.sqrt(p0))
    q_angles = np.full(n_qubits, q0, dtype=np.float64)

    initial_bits = repair_exact_budget(measure_solution(q_angles, rng), rng)
    initial_n = decode_acu_slots(initial_bits, n_sites)
    best_bits = initial_bits.copy()
    best_n = initial_n.copy()
    best_metrics = evaluator.evaluate(best_n)
    best_iteration = 0

    p_one0 = np.sin(q_angles) ** 2
    history: List[dict] = [{
        "t": 0,
        "F": best_metrics.fitness,
        "Phi_site": best_metrics.phi_site,
        "Phi_traf": best_metrics.phi_traf,
        "B": best_metrics.balance,
        "Ee": best_metrics.energy_eff,
        "E": best_metrics.total_energy,
        "CF": best_metrics.cloud_fallback,
        "theta_pi": theta / math.pi,
        "tabu_qubits": 0,
        "mean_p1": float(np.mean(p_one0)),
        "mean_qubit_uncertainty": float(np.mean(4.0 * p_one0 * (1.0 - p_one0))),
        "logical_evaluations": evaluator.logical_evaluations,
        "unique_evaluations": evaluator.unique_evaluations,
    }]

    if verbose_every > 0:
        print(
            f"[QTS Milan run {run_id:02d}] t={0:4d} | "
            f"F={best_metrics.fitness:.4f} | "
            f"Phi_s={best_metrics.phi_site:.3f} | "
            f"Phi_t={best_metrics.phi_traf:.3f} | "
            f"B={best_metrics.balance:.4f} | "
            f"Ee={best_metrics.energy_eff:.3f} | "
            f"CF={best_metrics.cloud_fallback} | initial feasible solution"
        )
    t_start = time.perf_counter()

    for t in range(1, iterations + 1):
        raw_neighborhood = measure_neighborhood(q_angles, neighborhood_size, rng)

        candidate_bits: List[np.ndarray] = []
        candidate_n: List[np.ndarray] = []
        candidate_metrics: List[FitnessResult] = []

        for k in range(neighborhood_size):
            bits = repair_exact_budget(raw_neighborhood[k], rng)
            n_vec = decode_acu_slots(bits, n_sites)
            metrics = evaluator.evaluate(n_vec)
            candidate_bits.append(bits)
            candidate_n.append(n_vec)
            candidate_metrics.append(metrics)

        best_idx = max(range(neighborhood_size), key=lambda i: metric_rank(candidate_metrics[i]))
        worst_idx = min(range(neighborhood_size), key=lambda i: metric_rank(candidate_metrics[i]))

        best_neighbor_bits = candidate_bits[best_idx]
        worst_neighbor_bits = candidate_bits[worst_idx]
        best_neighbor_n = candidate_n[best_idx]
        best_neighbor_metrics = candidate_metrics[best_idx]

        if metric_rank(best_neighbor_metrics) > metric_rank(best_metrics):
            best_bits = best_neighbor_bits.copy()
            best_n = best_neighbor_n.copy()
            best_metrics = best_neighbor_metrics
            best_iteration = t

        tabu_count = qts_move_gate_update(
            q_angles=q_angles,
            best_bits=best_neighbor_bits,
            worst_bits=worst_neighbor_bits,
            theta=theta,
        )

        p_one = np.sin(q_angles) ** 2
        mean_p1 = float(np.mean(p_one))
        mean_qubit_uncertainty = float(np.mean(4.0 * p_one * (1.0 - p_one)))

        history.append({
            "t": t,
            "F": best_metrics.fitness,
            "Phi_site": best_metrics.phi_site,
            "Phi_traf": best_metrics.phi_traf,
            "B": best_metrics.balance,
            "Ee": best_metrics.energy_eff,
            "E": best_metrics.total_energy,
            "CF": best_metrics.cloud_fallback,
            "theta_pi": theta / math.pi,
            "tabu_qubits": tabu_count,
            "mean_p1": mean_p1,
            "mean_qubit_uncertainty": mean_qubit_uncertainty,
            "logical_evaluations": evaluator.logical_evaluations,
            "unique_evaluations": evaluator.unique_evaluations,
        })

        if verbose_every > 0 and (t == 1 or t % verbose_every == 0 or t == iterations):
            print(
                f"[QTS Milan run {run_id:02d}] t={t:4d} | "
                f"F={best_metrics.fitness:.4f} | "
                f"Phi_s={best_metrics.phi_site:.3f} | "
                f"Phi_t={best_metrics.phi_traf:.3f} | "
                f"B={best_metrics.balance:.4f} | "
                f"Ee={best_metrics.energy_eff:.3f} | "
                f"CF={best_metrics.cloud_fallback} | "
                f"tabu={tabu_count}/{n_qubits}"
            )

    runtime = time.perf_counter() - t_start
    return QTSRunResult(
        run_id=run_id,
        seed=seed,
        best_iteration=best_iteration,
        best_bits=best_bits,
        best_n=best_n,
        best_metrics=best_metrics,
        history=history,
        logical_evaluations=evaluator.logical_evaluations,
        unique_evaluations=evaluator.unique_evaluations,
        runtime_sec=runtime,
    )


def write_history(path: Path, history: Sequence[dict]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_run_summary(path: Path, results: Sequence[QTSRunResult], evaluator: ESPPEvaluator) -> None:
    fields = [
        "run", "seed", "best_iteration", "F", "Phi_site", "Phi_traf", "B", "Ee",
        "E", "cloud_fallback", "active_sites", "mean_utilization",
        "logical_evaluations", "unique_evaluations", "runtime_sec",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            m = r.best_metrics
            writer.writerow({
                "run": r.run_id,
                "seed": r.seed,
                "best_iteration": r.best_iteration,
                "F": f"{m.fitness:.8f}",
                "Phi_site": f"{m.phi_site:.8f}",
                "Phi_traf": f"{m.phi_traf:.8f}",
                "B": f"{m.balance:.8f}",
                "Ee": f"{m.energy_eff:.8f}",
                "E": f"{m.total_energy:.8f}",
                "cloud_fallback": m.cloud_fallback,
                "active_sites": int(np.count_nonzero(r.best_n)),
                "mean_utilization": f"{evaluator.mean_utilization(r.best_n):.8f}",
                "logical_evaluations": r.logical_evaluations,
                "unique_evaluations": r.unique_evaluations,
                "runtime_sec": f"{r.runtime_sec:.6f}",
            })


def write_mean_convergence(path: Path, results: Sequence[QTSRunResult]) -> None:
    if not results:
        return
    keys = ["F", "Phi_site", "Phi_traf", "B", "Ee", "E", "CF"]
    n_t = min(len(r.history) for r in results)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["t"]
        for key in keys:
            fieldnames += [f"{key}_mean", f"{key}_std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(n_t):
            row = {"t": int(results[0].history[idx]["t"])}
            for key in keys:
                vals = np.asarray([r.history[idx][key] for r in results], dtype=float)
                row[f"{key}_mean"] = f"{float(np.mean(vals)):.8f}"
                row[f"{key}_std"] = f"{float(np.std(vals, ddof=0)):.8f}"
            writer.writerow(row)


def write_deployment_outputs(
    out_dir: Path,
    evaluator: ESPPEvaluator,
    result: QTSRunResult,
) -> None:
    bs = evaluator.bs
    n_vec = result.best_n
    sigma, min_dist = evaluator.service_mapping(n_vec)
    covered = sigma != -1

    loads = np.zeros(bs.n, dtype=np.float64)
    if np.any(covered):
        np.add.at(loads, sigma[covered], bs.workload[covered])

    acu_path = out_dir / "acu_deployment_QTS_Milan.csv"
    with acu_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "BS_ID", "Latitude", "Longitude", "ACU_Count", "Capacity",
            "ES_Workload", "ES_Utilization_pct",
        ])
        for i in np.flatnonzero(n_vec > 0):
            ci = evaluator.acu_capacity * int(n_vec[i])
            rho = loads[i] / ci if ci > 0 else 0.0
            writer.writerow([
                bs.bs_id[i], f"{bs.lat[i]:.6f}", f"{bs.lon[i]:.6f}", int(n_vec[i]),
                f"{ci:.8f}", f"{loads[i]:.8f}", f"{100.0 * rho:.2f}",
            ])

    alloc_path = out_dir / "bs_es_allocation_QTS_Milan.csv"
    with alloc_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "BS_ID", "Latitude", "Longitude", "Workload", "Local_ACU_Count",
            "Serving_ES_ID", "Serving_ES_ACU_Count", "Dist_to_ES_m",
        ])
        for k in range(bs.n):
            if sigma[k] != -1:
                sid = bs.bs_id[sigma[k]]
                serving_acus = int(n_vec[sigma[k]])
                dist = float(min_dist[k])
            else:
                sid = "CLOUD"
                serving_acus = 0
                dist = -1.0
            writer.writerow([
                bs.bs_id[k], f"{bs.lat[k]:.6f}", f"{bs.lon[k]:.6f}",
                f"{bs.workload[k]:.8f}", int(n_vec[k]), sid, serving_acus,
                f"{dist:.1f}",
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard QTS baseline for the VNEQTS Milan ESPP model."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("milan_bs_workload.csv"),
        help="Preprocessed Milan workload CSV (default: milan_bs_workload.csv)",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--neighbors", type=int, default=DEFAULT_NEIGHBORS,
        help="QTS neighborhood measurements per iteration (default: 50)",
    )
    parser.add_argument(
        "--theta-pi", type=float, default=DEFAULT_THETA_PI,
        help="Fixed rotation angle as a multiple of pi (default: 0.01)",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help="Independent runs; use 30 for paper-style robustness evaluation",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("qts_milan_results")
    )
    parser.add_argument(
        "--verbose-every", type=int, default=50,
        help="Print progress every N iterations; 0 disables progress lines",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.neighbors < 2:
        raise ValueError("--neighbors must be >= 2 so best/worst QTS references are defined")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.theta_pi <= 0:
        raise ValueError("--theta-pi must be > 0")
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Put milan_bs_workload.csv in the project directory or pass --data <path>."
        )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== QTS Baseline for Milan ESPP ==========")
    print(f"Data:          {args.data}")
    print(f"Iterations:    {args.iterations}")
    print(f"Measurements:  {args.neighbors} per iteration")
    print(f"Theta:         {args.theta_pi:.6f}*pi (fixed)")
    print(f"Runs:          {args.runs}")
    print(f"M / nmax:      {M_BUDGET} / {N_MAX}")
    print(f"ACU capacity:  {ACU_CAPACITY:.1f}")
    print(f"D_max:         {D_MAX:.1f} m")
    print("Fitness:       0.25*Phi_site + 0.25*Phi_traf + 0.25*B + 0.25*Ee")
    print("VNE features:  disabled (baseline)")
    print("Mutation:      disabled (standard QTS baseline)")
    print("Spectral escape: disabled (baseline)")

    bs = read_base_stations(args.data)
    print(f"BS count:      {bs.n}")
    print("Building distance matrix...")
    t0 = time.perf_counter()
    dist = build_distance_matrix(bs)
    print(f"Distance matrix ready in {time.perf_counter() - t0:.2f} s")

    evaluator = ESPPEvaluator(bs, dist)
    print(f"Total workload:{evaluator.total_workload:.8f}")
    print(
        f"Provisioned utilization: "
        f"{100.0 * evaluator.total_workload / (M_BUDGET * ACU_CAPACITY):.4f}%"
    )

    results: List[QTSRunResult] = []
    theta = args.theta_pi * math.pi

    for run_idx in range(args.runs):
        run_id = run_idx + 1
        run_seed = args.seed + run_idx
        evaluator.reset_counters_and_cache()

        print(f"\n--- Run {run_id}/{args.runs}, seed={run_seed} ---")
        result = run_qts_baseline(
            evaluator=evaluator,
            iterations=args.iterations,
            neighborhood_size=args.neighbors,
            theta=theta,
            seed=run_seed,
            run_id=run_id,
            verbose_every=args.verbose_every,
        )
        results.append(result)

        history_name = (
            "convergence_QTS_Milan.csv" if args.runs == 1
            else f"convergence_QTS_Milan_run{run_id:02d}.csv"
        )
        write_history(out_dir / history_name, result.history)

        m = result.best_metrics
        mean_util = evaluator.mean_utilization(result.best_n)
        print(
            f"Run {run_id} best: F={m.fitness:.6f}, "
            f"Phi_site={100*m.phi_site:.2f}%, Phi_traf={100*m.phi_traf:.2f}%, "
            f"B={m.balance:.6f}, Ee={m.energy_eff:.6f}, "
            f"CF={m.cloud_fallback}, active={np.count_nonzero(result.best_n)}, "
            f"mean_util={100*mean_util:.2f}%, best_t={result.best_iteration}, "
            f"runtime={result.runtime_sec:.2f}s"
        )

    write_run_summary(out_dir / "qts_milan_runs_summary.csv", results, evaluator)
    if len(results) > 1:
        write_mean_convergence(out_dir / "convergence_QTS_Milan_mean.csv", results)

    best_run = max(results, key=lambda r: metric_rank(r.best_metrics))
    write_deployment_outputs(out_dir, evaluator, best_run)

    final_metrics = best_run.best_metrics
    active_sites = int(np.count_nonzero(best_run.best_n))
    average_utilization = evaluator.mean_utilization(best_run.best_n)

    print(f"\nOutputs: {out_dir.resolve()}")
    print("\nFinal QTS Milan result")
    print(f"Fitness: {final_metrics.fitness:.8f}")
    print(f"Site coverage: {100.0 * final_metrics.phi_site:.8f}%")
    print(f"Traffic coverage: {100.0 * final_metrics.phi_traf:.8f}%")
    print(f"Load balance: {final_metrics.balance:.8f}")
    print(f"Energy efficiency: {final_metrics.energy_eff:.8f}")
    print(f"Total energy: {final_metrics.total_energy:.8f}")
    print(f"Cloud fallback BSs: {final_metrics.cloud_fallback} / {bs.n}")
    print(f"Active edge sites: {active_sites} / {bs.n}")
    print(f"Average utilization: {100.0 * average_utilization:.8f}%")


if __name__ == "__main__":
    main()
