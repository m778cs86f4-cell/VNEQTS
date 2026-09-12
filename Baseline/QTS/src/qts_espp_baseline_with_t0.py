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

DEFAULT_ITERATIONS = 500
DEFAULT_NEIGHBORS = 50
DEFAULT_THETA_PI = 0.01
DEFAULT_RUNS = 1
DEFAULT_SEED = 20260823
DEFAULT_INIT_MODE = "random"
DEFAULT_STRESS_CLUSTERS = 3
DEFAULT_STRESS_SELECTED_PROB = 0.65

D_MAX = 4800.0
M_BUDGET = 71
N_MAX = 3
T_OBS = 15_811_200.0            # 183 days, same as C++ implementation
B_EPS = 1e-9
CAPACITY_EPS = 1e-9

DEFAULT_WEIGHT_PROFILE = "A"
WEIGHT_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "A": (0.25, 0.25, 0.25, 0.25),
    "B": (0.15, 0.15, 0.20, 0.50),
    "C": (0.40, 0.40, 0.10, 0.10),
}

EARTH_R = 6_371_000.0


@dataclass(frozen=True)
class BaseStationData:
    bs_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    lambda_k: np.ndarray

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
    init_mode: str
    init_clusters: int
    coverage_target: Optional[float]
    coverage_tolerance: float
    fitness_cap: Optional[float]
    runtime_sec: float


def read_base_stations(filename: Path) -> BaseStationData:
    ids: List[str] = []
    lats: List[float] = []
    lons: List[float] = []
    lambdas: List[float] = []

    with filename.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # header
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {filename}") from exc

        for line_no, row in enumerate(reader, start=2):
            if not row or all(not x.strip() for x in row):
                continue
            if len(row) < 4:
                raise ValueError(
                    f"{filename}: line {line_no} has fewer than 4 columns: {row}"
                )
            try:
                bs_id = row[0].strip()
                lat = float(row[1])
                lon = float(row[2])
                total_duration = float(row[3])
            except ValueError as exc:
                raise ValueError(
                    f"{filename}: invalid numeric field at line {line_no}: {row[:4]}"
                ) from exc

            lam = total_duration / T_OBS
            if lam < 1e-6:
                lam = 1e-6

            ids.append(bs_id)
            lats.append(lat)
            lons.append(lon)
            lambdas.append(lam)

    if not ids:
        raise ValueError(f"No base-station records found in {filename}")

    return BaseStationData(
        bs_id=np.asarray(ids, dtype=object),
        lat=np.asarray(lats, dtype=np.float64),
        lon=np.asarray(lons, dtype=np.float64),
        lambda_k=np.asarray(lambdas, dtype=np.float64),
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
    def __init__(
        self,
        bs: BaseStationData,
        dist_matrix: np.ndarray,
        weight_profile: str = DEFAULT_WEIGHT_PROFILE,
    ):
        profile = weight_profile.upper()
        if profile not in WEIGHT_PROFILES:
            raise ValueError(
                f"Unknown weight profile {weight_profile!r}; "
                f"choose one of {sorted(WEIGHT_PROFILES)}."
            )
        weights = np.asarray(WEIGHT_PROFILES[profile], dtype=np.float64)
        if np.any(weights < 0.0) or not np.isclose(float(np.sum(weights)), 1.0):
            raise ValueError(f"Invalid weights for profile {profile}: {weights.tolist()}")

        self.bs = bs
        self.dist = dist_matrix
        self.weight_profile = profile
        self.weights = weights
        self.total_lambda = float(np.sum(bs.lambda_k))
        self.c0 =  self.total_lambda / M_BUDGET
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

        n_vec = np.asarray(n_vec, dtype=np.int8)
        active = np.flatnonzero(n_vec > 0)
        sigma = np.full(self.bs.n, -1, dtype=np.int32)
        min_dist = np.full(self.bs.n, np.inf, dtype=np.float32)

        if active.size == 0:
            return sigma, min_dist

        sub = self.dist[:, active]
        pos = np.argmin(sub, axis=1)
        mins = sub[np.arange(self.bs.n), pos]
        within_distance = mins <= D_MAX

        capacities = self.c0 * n_vec[active].astype(np.float64)
        remaining = capacities.copy()
        displaced_groups: List[np.ndarray] = []


        for active_pos, site in enumerate(active):
            assigned = np.flatnonzero(within_distance & (pos == active_pos))
            if assigned.size == 0:
                continue

            order = np.lexsort((assigned, sub[assigned, active_pos]))
            assigned = assigned[order]
            cumulative = np.cumsum(self.bs.lambda_k[assigned], dtype=np.float64)
            keep_count = int(
                np.searchsorted(
                    cumulative,
                    capacities[active_pos] + CAPACITY_EPS,
                    side="right",
                )
            )
            kept = assigned[:keep_count]
            if kept.size:
                sigma[kept] = int(site)
                min_dist[kept] = sub[kept, active_pos]
                remaining[active_pos] = max(
                    0.0,
                    remaining[active_pos] - float(np.sum(self.bs.lambda_k[kept])),
                )
            if keep_count < assigned.size:
                displaced_groups.append(assigned[keep_count:])

        if displaced_groups:
            displaced = np.concatenate(displaced_groups)
            reachable_count = np.sum(sub[displaced] <= D_MAX, axis=1)

            retry_order = np.lexsort(
                (displaced, -self.bs.lambda_k[displaced], reachable_count)
            )

            for station in displaced[retry_order]:
                demand = float(self.bs.lambda_k[station])
                feasible = (
                    (sub[station] <= D_MAX)
                    & (remaining + CAPACITY_EPS >= demand)
                )
                feasible_pos = np.flatnonzero(feasible)
                if feasible_pos.size == 0:
                    continue
                chosen_pos = int(
                    feasible_pos[np.argmin(sub[station, feasible_pos])]
                )
                sigma[station] = int(active[chosen_pos])
                min_dist[station] = sub[station, chosen_pos]
                remaining[chosen_pos] = max(
                    0.0, remaining[chosen_pos] - demand
                )

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
        covered_lambda = float(np.sum(self.bs.lambda_k[covered]))
        cloud_count = self.bs.n - covered_sites

        phi_site = covered_sites / self.bs.n
        phi_traf = covered_lambda / self.total_lambda if self.total_lambda > 0 else 0.0

        loads = np.zeros(self.bs.n, dtype=np.float64)
        if covered_sites:
            np.add.at(loads, sigma[covered], self.bs.lambda_k[covered])

        active = np.flatnonzero(n_vec > 0)
        if active.size:
            capacity = self.c0 * n_vec[active].astype(np.float64)
            overload = loads[active] - capacity
            tolerance = CAPACITY_EPS * np.maximum(1.0, capacity)
            if np.any(overload > tolerance):
                worst = float(np.max(overload))
                raise RuntimeError(
                    f"Hard capacity constraint violated; maximum overload={worst:.12g}"
                )
            rho = np.divide(
                loads[active], capacity,
                out=np.zeros_like(capacity),
                where=capacity > 0,
            )
            rho = np.clip(rho, 0.0, 1.0)
            total_energy = float(
                np.sum(n_vec[active].astype(np.float64) * (1.0 + rho))
            )

            if active.size >= 2:
                mean_rho = float(np.mean(rho))
                std_rho = float(np.std(rho, ddof=0))
                cv = std_rho / (mean_rho + B_EPS)
                balance = 1.0 - min(1.0, cv)
            else:
                balance = 1.0
        else:
            total_energy = 0.0
            balance = 0.0

        energy_eff = 1.0 - (total_energy - M_BUDGET) / M_BUDGET
        energy_eff = float(np.clip(energy_eff, 0.0, 1.0))

        w_site, w_traf, w_balance, w_energy = self.weights
        fitness = (
            w_site * phi_site
            + w_traf * phi_traf
            + w_balance * balance
            + w_energy * energy_eff
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
            np.add.at(loads, sigma[covered], self.bs.lambda_k[covered])

        active = np.flatnonzero(n_vec > 0)
        if active.size == 0:
            return 0.0
        capacity = self.c0 * n_vec[active].astype(np.float64)
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


def deployment_to_bits(
    n_vec: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n_vec = np.asarray(n_vec, dtype=np.int8)
    bits = np.zeros(n_vec.size * N_MAX, dtype=np.bool_)

    for site, count_raw in enumerate(n_vec):
        count = int(count_raw)
        if count < 0 or count > N_MAX:
            raise ValueError(
                f"Invalid ACU count n[{site}]={count}; expected 0..{N_MAX}."
            )
        if count:
            slots = rng.choice(N_MAX, size=count, replace=False)
            bits[site * N_MAX + slots] = True

    if int(np.count_nonzero(bits)) != M_BUDGET:
        raise RuntimeError("deployment_to_bits produced a non-budget-feasible encoding.")
    return bits


def initialize_qubits_around_solution(
    initial_bits: np.ndarray,
    p_selected: float,
) -> np.ndarray:
    if not (0.5 < p_selected < 1.0):
        raise ValueError("stress initialization probability must lie in (0.5, 1.0)")

    initial_bits = np.asarray(initial_bits, dtype=np.bool_)
    selected_count = int(np.count_nonzero(initial_bits))
    if selected_count != M_BUDGET:
        raise ValueError(
            f"Expected exactly {M_BUDGET} selected slots, got {selected_count}."
        )

    unselected_count = initial_bits.size - selected_count
    p_unselected = (
        M_BUDGET - selected_count * p_selected
    ) / unselected_count
    if not (0.0 < p_unselected < 0.5):
        raise ValueError(
            f"Derived unselected-slot probability is invalid: {p_unselected}."
        )

    q_angles = np.empty(initial_bits.size, dtype=np.float64)
    q_angles[initial_bits] = math.asin(math.sqrt(p_selected))
    q_angles[~initial_bits] = math.asin(math.sqrt(p_unselected))
    return q_angles


def clustered_stress_deployment(
    dist_matrix: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_sites = int(dist_matrix.shape[0])
    min_active = math.ceil(M_BUDGET / N_MAX)
    if n_sites < min_active:
        raise ValueError(
            f"Need at least {min_active} sites for a feasible deployment; got {n_sites}."
        )
    if not (2 <= n_clusters <= min(3, min_active, n_sites)):
        raise ValueError("clustered-stress initialization requires 2 or 3 clusters")

    centers = [int(rng.integers(n_sites))]
    min_center_separation = 2.0 * D_MAX
    while len(centers) < n_clusters:
        eligible = np.ones(n_sites, dtype=np.bool_)
        eligible[np.asarray(centers, dtype=np.int32)] = False
        center_dist = dist_matrix[:, np.asarray(centers, dtype=np.int32)]
        eligible &= np.min(center_dist, axis=1) >= min_center_separation
        candidates = np.flatnonzero(eligible)
        if candidates.size == 0:
            candidates = np.setdiff1d(
                np.arange(n_sites, dtype=np.int32),
                np.asarray(centers, dtype=np.int32),
                assume_unique=False,
            )
        centers.append(int(rng.choice(candidates)))

    quotas = np.full(n_clusters, min_active // n_clusters, dtype=np.int32)
    quotas[: min_active % n_clusters] += 1
    selected_mask = np.zeros(n_sites, dtype=np.bool_)
    selected_sites: List[int] = []

    for center, quota in zip(centers, quotas):
        order = np.argsort(dist_matrix[center], kind="stable")
        available = order[~selected_mask[order]]
        chosen = available[: int(quota)]
        selected_mask[chosen] = True
        selected_sites.extend(int(site) for site in chosen)

    selected = np.asarray(selected_sites, dtype=np.int32)
    if selected.size != min_active:
        raise RuntimeError("Failed to construct the requested clustered deployment.")

    n_vec = np.zeros(n_sites, dtype=np.int8)
    base = M_BUDGET // min_active
    remainder = M_BUDGET % min_active
    n_vec[selected] = base
    if remainder:
        extra_sites = rng.choice(selected, size=remainder, replace=False)
        n_vec[extra_sites] += 1

    if int(np.sum(n_vec)) != M_BUDGET or int(np.max(n_vec)) > N_MAX:
        raise RuntimeError("Clustered stress initialization violated the ACU constraints.")
    return n_vec, np.asarray(centers, dtype=np.int32)


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


def coverage_operating_violation(
    m: FitnessResult,
    target: Optional[float],
    tolerance: float,
    fitness_cap: Optional[float],
) -> float:
    if target is None:
        return 0.0
    lower = target - tolerance
    upper = target + tolerance

    def band_violation(value: float) -> float:
        return max(0.0, lower - value, value - upper)

    violation = band_violation(m.phi_site) + band_violation(m.phi_traf)
    if fitness_cap is not None:
        violation += max(0.0, m.fitness - fitness_cap)
    return float(violation)


def metric_rank(
    m: FitnessResult,
    coverage_target: Optional[float] = None,
    coverage_tolerance: float = 0.0,
    fitness_cap: Optional[float] = None,
) -> Tuple[float, ...]:
    if coverage_target is not None:
        violation = coverage_operating_violation(
            m, coverage_target, coverage_tolerance, fitness_cap
        )
        target_error = (
            abs(m.phi_site - coverage_target)
            + abs(m.phi_traf - coverage_target)
        )
        return (
            -violation,
            -target_error,
            m.energy_eff,
            m.balance,
            m.fitness,
            -m.total_energy,
            -m.cloud_fallback,
        )

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
    init_mode: str = DEFAULT_INIT_MODE,
    stress_clusters: int = DEFAULT_STRESS_CLUSTERS,
    stress_selected_prob: float = DEFAULT_STRESS_SELECTED_PROB,
    coverage_target: Optional[float] = None,
    coverage_tolerance: float = 0.0,
    fitness_cap: Optional[float] = None,
    run_id: int = 1,
    verbose_every: int = 0,
) -> QTSRunResult:
    rng = np.random.default_rng(seed)
    n_sites = evaluator.bs.n
    n_qubits = n_sites * N_MAX

    if init_mode == "random":
        # Standard constraint-aware random initialization.
        p0 = M_BUDGET / n_qubits
        q0 = math.asin(math.sqrt(p0))
        q_angles = np.full(n_qubits, q0, dtype=np.float64)
        initial_bits = repair_exact_budget(measure_solution(q_angles, rng), rng)
        initial_n = decode_acu_slots(initial_bits, n_sites)
        center_ids = np.empty(0, dtype=np.int32)
    elif init_mode == "clustered-stress":
        initial_n, center_ids = clustered_stress_deployment(
            evaluator.dist,
            n_clusters=stress_clusters,
            rng=rng,
        )
        initial_bits = deployment_to_bits(initial_n, rng)
        q_angles = initialize_qubits_around_solution(
            initial_bits,
            p_selected=stress_selected_prob,
        )
    else:
        raise ValueError(f"Unknown initialization mode: {init_mode}")

    best_bits = initial_bits.copy()
    best_n = initial_n.copy()
    best_metrics = evaluator.evaluate(best_n)
    best_iteration = 0

    def rank(metrics: FitnessResult) -> Tuple[float, ...]:
        return metric_rank(
            metrics,
            coverage_target=coverage_target,
            coverage_tolerance=coverage_tolerance,
            fitness_cap=fitness_cap,
        )

    history: List[dict] = []

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

        best_idx = max(range(neighborhood_size), key=lambda i: rank(candidate_metrics[i]))
        worst_idx = min(range(neighborhood_size), key=lambda i: rank(candidate_metrics[i]))

        best_neighbor_bits = candidate_bits[best_idx]
        worst_neighbor_bits = candidate_bits[worst_idx]
        best_neighbor_n = candidate_n[best_idx]
        best_neighbor_metrics = candidate_metrics[best_idx]

        if rank(best_neighbor_metrics) > rank(best_metrics):
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
            "init_mode": init_mode,
            "init_clusters": int(center_ids.size),
            "weight_profile": evaluator.weight_profile,
            "w_site": float(evaluator.weights[0]),
            "w_traf": float(evaluator.weights[1]),
            "w_balance": float(evaluator.weights[2]),
            "w_energy": float(evaluator.weights[3]),
            "coverage_target": "" if coverage_target is None else coverage_target,
            "coverage_tolerance": coverage_tolerance,
            "fitness_cap": "" if fitness_cap is None else fitness_cap,
            "operating_violation": coverage_operating_violation(
                best_metrics, coverage_target, coverage_tolerance, fitness_cap
            ),
        })
        if t % 10 != 0:
            history.pop()

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
        init_mode=init_mode,
        init_clusters=int(center_ids.size),
        coverage_target=coverage_target,
        coverage_tolerance=coverage_tolerance,
        fitness_cap=fitness_cap,
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
        "run", "seed", "init_mode", "init_clusters", "weight_profile",
        "w_site", "w_traf", "w_balance", "w_energy",
        "coverage_target", "coverage_tolerance", "fitness_cap",
        "operating_violation", "best_iteration",
        "F", "Phi_site", "Phi_traf", "B", "Ee",
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
                "init_mode": r.init_mode,
                "init_clusters": r.init_clusters,
                "weight_profile": evaluator.weight_profile,
                "w_site": f"{evaluator.weights[0]:.2f}",
                "w_traf": f"{evaluator.weights[1]:.2f}",
                "w_balance": f"{evaluator.weights[2]:.2f}",
                "w_energy": f"{evaluator.weights[3]:.2f}",
                "coverage_target": (
                    "" if r.coverage_target is None else f"{r.coverage_target:.8f}"
                ),
                "coverage_tolerance": f"{r.coverage_tolerance:.8f}",
                "fitness_cap": (
                    "" if r.fitness_cap is None else f"{r.fitness_cap:.8f}"
                ),
                "operating_violation": f"{coverage_operating_violation(m, r.coverage_target, r.coverage_tolerance, r.fitness_cap):.8f}",
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
            # Preserve the stored labels t=10,20,... rather than reconstructing
            # them from the row index.
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
        np.add.at(loads, sigma[covered], bs.lambda_k[covered])

    acu_path = out_dir / "acu_deployment_QTS.csv"
    with acu_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "BS_ID", "Latitude", "Longitude", "ACU_Count", "Capacity",
            "ES_Lambda", "ES_Utilization_pct",
        ])
        for i in np.flatnonzero(n_vec > 0):
            ci = evaluator.c0 * int(n_vec[i])
            rho = min(loads[i] / ci, 1.0) if ci > 0 else 0.0
            writer.writerow([
                bs.bs_id[i], f"{bs.lat[i]:.6f}", f"{bs.lon[i]:.6f}", int(n_vec[i]),
                f"{ci:.4f}", f"{loads[i]:.4f}", f"{100.0 * rho:.2f}",
            ])

    alloc_path = out_dir / "bs_es_allocation_QTS.csv"
    with alloc_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "BS_ID", "Latitude", "Longitude", "Lambda", "ACU_Count",
            "Serving_ES_ID", "Dist_to_ES_m",
        ])
        for k in range(bs.n):
            if sigma[k] != -1:
                sid = bs.bs_id[sigma[k]]
                dist = float(min_dist[k])
            else:
                sid = "CLOUD"
                dist = -1.0
            writer.writerow([
                bs.bs_id[k], f"{bs.lat[k]:.6f}", f"{bs.lon[k]:.6f}",
                f"{bs.lambda_k[k]:.10f}", int(n_vec[k]), sid, f"{dist:.1f}",
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard QTS baseline adapted to VNEQTS's ACU-based ESPP model."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("bs_statistics_all_12files.csv"),
        help="Shanghai Telecom BS statistics CSV (default: bs_statistics_all_12files.csv)",
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
        "--weight-profile",
        type=str.upper,
        choices=tuple(WEIGHT_PROFILES),
        default=DEFAULT_WEIGHT_PROFILE,
        help=(
            "Fitness weighting profile: A=[0.25,0.25,0.25,0.25], "
            "B=[0.15,0.15,0.20,0.50] energy-prioritized, or "
            "C=[0.40,0.40,0.10,0.10] coverage-prioritized."
        ),
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=None,
        help=(
            "Optional disclosed operating-point target applied to both Phi_site "
            "and Phi_traf. Omit for the standard unconstrained QTS baseline."
        ),
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=0.01,
        help="Half-width of the coverage target band (default: 0.01).",
    )
    parser.add_argument(
        "--fitness-cap",
        type=float,
        default=None,
        help=(
            "Optional raw-fitness upper bound used only with --coverage-target; "
            "reported fitness is never clipped or rewritten."
        ),
    )
    parser.add_argument(
        "--init-mode",
        choices=("random", "clustered-stress"),
        default=DEFAULT_INIT_MODE,
        help=(
            "Initialization policy. 'random' is the standard baseline; "
            "'clustered-stress' deliberately starts from a poor geographically "
            "clustered feasible deployment and must be reported as a stress test."
        ),
    )
    parser.add_argument(
        "--stress-clusters",
        type=int,
        choices=(2, 3),
        default=DEFAULT_STRESS_CLUSTERS,
        help="Number of geographic clusters for --init-mode clustered-stress.",
    )
    parser.add_argument(
        "--stress-selected-prob",
        type=float,
        default=DEFAULT_STRESS_SELECTED_PROB,
        help=(
            "Initial P(1) on clustered solution slots; must be in (0.5, 1.0). "
            "Lower values permit faster exploration (default: 0.65)."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Output directory. If omitted, a separate directory is selected from "
            "the initialization mode and weight profile."
        ),
    )
    parser.add_argument(
        "--verbose-every", type=int, default=0,
        help="Retained for command compatibility; iterative terminal logs are disabled.",
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
    if args.coverage_target is not None and not (0.0 < args.coverage_target < 1.0):
        raise ValueError("--coverage-target must lie in (0, 1)")
    if not (0.0 <= args.coverage_tolerance < 0.5):
        raise ValueError("--coverage-tolerance must lie in [0, 0.5)")
    if args.fitness_cap is not None and args.coverage_target is None:
        raise ValueError("--fitness-cap requires --coverage-target")
    if args.fitness_cap is not None and not (0.0 < args.fitness_cap <= 1.0):
        raise ValueError("--fitness-cap must lie in (0, 1]")
    if not (0.5 < args.stress_selected_prob < 1.0):
        raise ValueError("--stress-selected-prob must lie in (0.5, 1.0)")
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Put bs_statistics_all_12files.csv in the project directory or pass --data <path>."
        )

    if args.output_dir is not None:
        out_dir = args.output_dir
    elif args.init_mode == "clustered-stress":
        out_dir = Path(f"qts_clustered_stress_profile_{args.weight_profile}_results")
    else:
        out_dir = Path(f"qts_weight_profile_{args.weight_profile}_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data:          {args.data.name}")
    print(f"Iterations:    {args.iterations}")
    print(f"Measurements:  {args.neighbors}")
    print(f"Theta:         {args.theta_pi:.6f}*pi")
    weights = WEIGHT_PROFILES[args.weight_profile]
    print(f"Weight profile:{args.weight_profile}")
    print(
        "Weights:       "
        f"Phi_site={weights[0]:.2f}, Phi_traf={weights[1]:.2f}, "
        f"B={weights[2]:.2f}, Ee={weights[3]:.2f}"
    )
    print(f"Init mode:     {args.init_mode}")
    if args.coverage_target is None:
        print("Operating mode:standard unconstrained fitness maximization")
    else:
        cap_text = "none" if args.fitness_cap is None else f"{args.fitness_cap:.4f}"
        print(
            "Operating mode:coverage target "
            f"{100*args.coverage_target:.2f}% +/- {100*args.coverage_tolerance:.2f}%, "
            f"raw F cap={cap_text} (disclosed constrained experiment)"
        )
    print(f"Runs:          {args.runs}")
    print(f"M / nmax:      {M_BUDGET} / {N_MAX}")

    bs = read_base_stations(args.data)
    print(f"BS count:      {bs.n}")
    print("Building distance matrix...")
    dist = build_distance_matrix(bs)

    evaluator = ESPPEvaluator(bs, dist, weight_profile=args.weight_profile)
    print(f"Sum lambda:    {evaluator.total_lambda:.6f}")

    results: List[QTSRunResult] = []
    theta = args.theta_pi * math.pi

    for run_idx in range(args.runs):
        run_id = run_idx + 1
        run_seed = args.seed + run_idx
        evaluator.reset_counters_and_cache()

        result = run_qts_baseline(
            evaluator=evaluator,
            iterations=args.iterations,
            neighborhood_size=args.neighbors,
            theta=theta,
            seed=run_seed,
            init_mode=args.init_mode,
            stress_clusters=args.stress_clusters,
            stress_selected_prob=args.stress_selected_prob,
            coverage_target=args.coverage_target,
            coverage_tolerance=args.coverage_tolerance,
            fitness_cap=args.fitness_cap,
            run_id=run_id,
            verbose_every=args.verbose_every,
        )
        results.append(result)

        history_name = (
            "convergence_QTS.csv" if args.runs == 1
            else f"convergence_QTS_run{run_id:02d}.csv"
        )
        write_history(out_dir / history_name, result.history)


    write_run_summary(out_dir / "qts_runs_summary.csv", results, evaluator)
    if len(results) > 1:
        write_mean_convergence(out_dir / "convergence_QTS_mean.csv", results)

    # Deployment/allocation files use the best run, whereas paper comparison
    # statistics should use the mean over independent runs.
    best_run = max(
        results,
        key=lambda r: metric_rank(
            r.best_metrics,
            coverage_target=args.coverage_target,
            coverage_tolerance=args.coverage_tolerance,
            fitness_cap=args.fitness_cap,
        ),
    )
    write_deployment_outputs(out_dir, evaluator, best_run)

    f_vals = np.asarray([r.best_metrics.fitness for r in results], dtype=float)
    phi_s = np.asarray([r.best_metrics.phi_site for r in results], dtype=float)
    phi_t = np.asarray([r.best_metrics.phi_traf for r in results], dtype=float)
    b_vals = np.asarray([r.best_metrics.balance for r in results], dtype=float)
    ee_vals = np.asarray([r.best_metrics.energy_eff for r in results], dtype=float)
    util_vals = np.asarray([evaluator.mean_utilization(r.best_n) for r in results], dtype=float)
    cf_vals = np.asarray([r.best_metrics.cloud_fallback for r in results], dtype=float)

    print()
    print(f"F:             {np.mean(f_vals):.6f} +/- {np.std(f_vals):.6f}")
    print(f"Phi_site:      {100*np.mean(phi_s):.4f}%")
    print(f"Phi_traf:      {100*np.mean(phi_t):.4f}%")
    print(f"B:             {np.mean(b_vals):.6f}")
    print(f"Ee:            {np.mean(ee_vals):.6f}")
    print()
    print(f"Mean util.:    {100*np.mean(util_vals):.4f}%")


if __name__ == "__main__":
    main()
