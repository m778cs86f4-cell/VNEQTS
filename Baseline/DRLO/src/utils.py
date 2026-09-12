import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


T_OBS_SECONDS = 183.0 * 24.0 * 3600.0
DEFAULT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
EPS = 1e-9


def cal_distance(lat_a, lng_a, lat_b, lng_b):
    radius_m = 6371000.0
    lat1 = math.radians(float(lat_a))
    lat2 = math.radians(float(lat_b))
    d_lat = lat2 - lat1
    d_lng = math.radians(float(lng_b) - float(lng_a))
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _distance_matrix_m(latitudes: np.ndarray, longitudes: np.ndarray, block_size: int = 512) -> np.ndarray:
    lat = np.radians(latitudes.astype(np.float64))
    lon = np.radians(longitudes.astype(np.float64))
    n = len(lat)
    dist = np.empty((n, n), dtype=np.float32)
    radius_m = 6371000.0

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        lat1 = lat[start:end, None]
        lon1 = lon[start:end, None]
        d_lat = lat[None, :] - lat1
        d_lon = lon[None, :] - lon1
        a = (
            np.sin(d_lat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat[None, :]) * np.sin(d_lon / 2.0) ** 2
        )
        dist[start:end, :] = (radius_m * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))).astype(np.float32)

    np.fill_diagonal(dist, 0.0)
    return dist


def _column_index(columns: Iterable, candidates: Tuple[str, ...], fallback: int) -> int:
    normalized = [str(c).strip().lower() for c in columns]
    for candidate in candidates:
        candidate = candidate.lower()
        for idx, col in enumerate(normalized):
            if col == candidate or candidate in col:
                return idx
    return fallback


def resolve_bs_num(data_path: str, requested_bs_num: int) -> int:
    df = pd.read_csv(data_path, usecols=[0])
    available = len(df)
    if requested_bs_num is None or int(requested_bs_num) <= 0 or int(requested_bs_num) > available:
        return available
    return int(requested_bs_num)


def load_base_stations(data_path: str, bs_num: int) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if bs_num is not None and int(bs_num) > 0:
        df = df.head(int(bs_num)).copy()

    id_idx = _column_index(df.columns, ("bs_id", "base_station", "基站id", "基站"), 0)
    lat_idx = _column_index(df.columns, ("latitude", "lat", "纬度"), 1)
    lon_idx = _column_index(df.columns, ("longitude", "lon", "lng", "经度"), 2)
    duration_idx = _column_index(df.columns, ("total_duration", "duration", "停留"), 3)
    lambda_idx = _column_index(df.columns, ("lambda", "lambda_k", "workload", "mean_internet"), -1)

    result = pd.DataFrame()
    result["bs_id"] = df.iloc[:, id_idx].astype(str)
    result["lat"] = pd.to_numeric(df.iloc[:, lat_idx], errors="coerce")
    result["lon"] = pd.to_numeric(df.iloc[:, lon_idx], errors="coerce")

    duration = pd.to_numeric(df.iloc[:, duration_idx], errors="coerce")
    lambda_values = pd.to_numeric(df.iloc[:, lambda_idx], errors="coerce") if lambda_idx >= 0 else None
    if lambda_values is not None and lambda_values.notna().any():
        result["lambda_k"] = lambda_values
    elif duration.notna().any():
        result["lambda_k"] = duration / T_OBS_SECONDS
    else:
        raise ValueError(
            "Cannot find total_duration or lambda column in base-station CSV. "
            "Expected bs_v2.cpp output: bs_id,latitude,longitude,total_duration(s),unique_users,lambda."
        )

    result = result.dropna(subset=["lat", "lon", "lambda_k"]).reset_index(drop=True)
    result["lambda_k"] = result["lambda_k"].clip(lower=1e-6)
    return result


def repair_deployment(
    deployment: Iterable,
    acu_budget: int,
    n_max: int,
    scores: Optional[Iterable] = None,
) -> np.ndarray:
    allocation = np.asarray(deployment, dtype=np.int64).ravel().copy()
    if allocation.size == 0:
        return allocation

    acu_budget = int(min(max(0, acu_budget), allocation.size * int(n_max)))
    n_max = int(max(1, n_max))
    allocation = np.clip(allocation, 0, n_max)

    if scores is None:
        score_arr = np.zeros(allocation.size, dtype=np.float64)
        tie_break = np.random.random(allocation.size)
    else:
        score_arr = np.asarray(scores, dtype=np.float64).ravel()
        if score_arr.size != allocation.size:
            score_arr = np.resize(score_arr, allocation.size)
        score_arr = np.nan_to_num(score_arr, nan=0.0, posinf=0.0, neginf=0.0)
        tie_break = np.linspace(0.0, 1e-12, allocation.size)

    total = int(allocation.sum())
    while total > acu_budget:
        candidates = np.flatnonzero(allocation > 0)
        if candidates.size == 0:
            break
        chosen = candidates[np.lexsort((tie_break[candidates], score_arr[candidates]))[0]]
        allocation[chosen] -= 1
        total -= 1

    while total < acu_budget:
        candidates = np.flatnonzero(allocation < n_max)
        if candidates.size == 0:
            break
        chosen = candidates[np.lexsort((tie_break[candidates], -score_arr[candidates]))[0]]
        allocation[chosen] += 1
        total += 1

    return allocation


def random_deployment(bs_num: int, acu_budget: int, n_max: int) -> np.ndarray:
    allocation = np.zeros(int(bs_num), dtype=np.int64)
    capacity = int(bs_num) * int(n_max)
    acu_budget = int(min(max(0, acu_budget), capacity))
    placed = 0
    while placed < acu_budget:
        idx = random.randrange(int(bs_num))
        if allocation[idx] < int(n_max):
            allocation[idx] += 1
            placed += 1
    return allocation


def transform_action(output, acu_budget: int, n_max: int = 1) -> np.ndarray:
    """Project actor output to an integer ACU deployment vector."""
    scores = np.asarray(output, dtype=np.float64).ravel()
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.size == 0:
        return np.array([], dtype=np.int64)

    lo = float(scores.min())
    hi = float(scores.max())
    if abs(hi - lo) < EPS:
        allocation = np.zeros(scores.size, dtype=np.int64)
        order = np.argsort(-scores)
        for idx in order[: min(int(acu_budget), scores.size)]:
            allocation[idx] = 1
    else:
        scaled = (scores - lo) / (hi - lo)
        allocation = np.rint(scaled * int(n_max)).astype(np.int64)

    return repair_deployment(allocation, int(acu_budget), int(n_max), scores)


@dataclass
class EvaluationResult:
    fitness: float
    phi_site: float
    phi_traf: float
    balance: float
    energy_eff: float
    total_energy: float
    cloud_fallback: int
    active_sites: int
    avg_util: float
    std_util: float
    max_util: float
    overload_count: int
    avg_distance_m: float

    def as_state(self, state_dim: int, acu_budget: int, bs_num: int, d_max_m: float) -> np.ndarray:
        base = np.array(
            [
                self.fitness,
                self.phi_site,
                self.phi_traf,
                self.balance,
                self.energy_eff,
                self.total_energy / max(2.0 * acu_budget, EPS),
                self.cloud_fallback / max(bs_num, 1),
                self.active_sites / max(bs_num, 1),
                self.avg_util,
                self.std_util,
                self.max_util,
                self.overload_count / max(self.active_sites, 1),
                min(self.avg_distance_m / max(d_max_m, EPS), 1.0),
            ],
            dtype=np.float64,
        )
        if state_dim <= base.size:
            return base[:state_dim]
        repeats = int(math.ceil(state_dim / base.size))
        return np.tile(base, repeats)[:state_dim]

    def to_dict(self) -> Dict[str, float]:
        return {
            "fitness": self.fitness,
            "phi_site": self.phi_site,
            "phi_traf": self.phi_traf,
            "balance": self.balance,
            "energy_eff": self.energy_eff,
            "total_energy": self.total_energy,
            "cloud_fallback": self.cloud_fallback,
            "active_sites": self.active_sites,
            "avg_util": self.avg_util,
            "std_util": self.std_util,
            "max_util": self.max_util,
            "overload_count": self.overload_count,
            "avg_distance_m": self.avg_distance_m,
        }


class Assign:
    def __init__(
        self,
        bs_num,
        acu_budget,
        data_path,
        state_dim,
        init_vector,
        n_max=3,
        d_max_m=1500.0,
        weights=None,
        service_capacity=11000.0,
        capacity_constrained_mapping=False,
        site_idle_energy=0.0,
        acu_static_energy=1.0,
        acu_dynamic_energy=1.0,
        dynamic_energy_exponent=1.0,
        energy_efficiency_bounds="vneqts",
        load_balance_positive_only=True,
    ):
        self.bs_num = int(bs_num)
        self.acu_budget = int(acu_budget)
        self.es_num = self.acu_budget
        self.data_path = data_path
        self.state_dim = int(state_dim)
        self.days = self.state_dim
        self.n_max = int(n_max)
        self.d_max_m = float(d_max_m)
        self.weights = self._normalise_weights(weights)
        self.capacity_constrained_mapping = bool(capacity_constrained_mapping)
        self.site_idle_energy = float(site_idle_energy)
        self.acu_static_energy = float(acu_static_energy)
        self.acu_dynamic_energy = float(acu_dynamic_energy)
        self.dynamic_energy_exponent = float(dynamic_energy_exponent)
        self.energy_efficiency_bounds = str(energy_efficiency_bounds)
        self.load_balance_positive_only = bool(load_balance_positive_only)

        bs_df = load_base_stations(data_path, self.bs_num)
        self.bs_num = len(bs_df)
        self.bs_data = bs_df.to_dict("records")
        self.bs_ids = bs_df["bs_id"].to_numpy()
        self.latitudes = bs_df["lat"].to_numpy(dtype=np.float64)
        self.longitudes = bs_df["lon"].to_numpy(dtype=np.float64)
        self.lambda_k = bs_df["lambda_k"].to_numpy(dtype=np.float64)
        self.total_lambda = float(self.lambda_k.sum())
        self.service_capacity = (
            float(service_capacity)
            if service_capacity is not None and float(service_capacity) > 0.0
            else self.total_lambda  / max(self.acu_budget, 1)
        )

        self.dist_matrix = _distance_matrix_m(self.latitudes, self.longitudes)
        self.station_order_by_workload = np.argsort(-self.lambda_k, kind="stable")
        self.init_vector = self._coerce_initial_deployment(init_vector)
        self.last_metrics = None

    def _normalise_weights(self, weights) -> Tuple[float, float, float, float]:
        if weights is None:
            weights = DEFAULT_WEIGHTS
        if isinstance(weights, dict):
            weights = (
                weights.get("site", weights.get("phi_site", 0.25)),
                weights.get("traffic", weights.get("phi_traf", 0.25)),
                weights.get("balance", 0.25),
                weights.get("energy", weights.get("energy_eff", 0.25)),
            )
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.size != 4 or float(weights.sum()) <= 0.0:
            weights = np.asarray(DEFAULT_WEIGHTS, dtype=np.float64)
        weights = weights / weights.sum()
        return tuple(float(x) for x in weights)

    def _coerce_initial_deployment(self, init_vector) -> np.ndarray:
        arr = np.asarray(init_vector, dtype=np.int64).ravel()
        if arr.size == self.bs_num:
            return repair_deployment(arr, self.acu_budget, self.n_max)

        allocation = np.zeros(self.bs_num, dtype=np.int64)
        for idx in arr:
            if 0 <= int(idx) < self.bs_num and allocation[int(idx)] < self.n_max:
                allocation[int(idx)] += 1
        if int(allocation.sum()) == 0:
            return random_deployment(self.bs_num, self.acu_budget, self.n_max)
        return repair_deployment(allocation, self.acu_budget, self.n_max)

    def get_init_action(self):
        return self.init_vector.copy()

    def compute_service_mapping(self, deployment: Iterable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = repair_deployment(deployment, self.acu_budget, self.n_max)
        active = np.flatnonzero(n > 0)
        sigma = np.full(self.bs_num, -1, dtype=np.int64)
        site_loads = np.zeros(self.bs_num, dtype=np.float64)

        if active.size == 0:
            return sigma, site_loads, n

        if self.capacity_constrained_mapping:
            for bs_idx in self.station_order_by_workload:
                best_site = -1
                best_distance = float("inf")
                workload = self.lambda_k[bs_idx]
                for site in active:
                    distance = float(self.dist_matrix[site, bs_idx])
                    capacity = float(n[site]) * self.service_capacity
                    has_capacity = site_loads[site] + workload <= capacity + EPS
                    if (
                        distance <= self.d_max_m
                        and has_capacity
                        and (distance < best_distance or (distance == best_distance and site < best_site))
                    ):
                        best_distance = distance
                        best_site = int(site)

                if best_site != -1:
                    sigma[bs_idx] = best_site
                    site_loads[best_site] += workload
            return sigma, site_loads, n

        active_dist = self.dist_matrix[:, active]
        nearest_pos = np.argmin(active_dist, axis=1)
        nearest_dist = active_dist[np.arange(self.bs_num), nearest_pos]
        covered = nearest_dist <= self.d_max_m
        sigma[covered] = active[nearest_pos[covered]]
        np.add.at(site_loads, sigma[covered], self.lambda_k[covered])
        return sigma, site_loads, n

    def evaluate_system(self, deployment: Iterable, return_details: bool = False):
        sigma, site_loads, n = self.compute_service_mapping(deployment)
        active = np.flatnonzero(n > 0)
        covered = sigma != -1
        covered_sites = int(covered.sum())
        covered_lambda = float(self.lambda_k[covered].sum())

        phi_site = covered_sites / max(self.bs_num, 1)
        phi_traf = covered_lambda / max(self.total_lambda, EPS)
        cloud_fallback = self.bs_num - covered_sites

        if active.size == 0:
            result = EvaluationResult(0.0, phi_site, phi_traf, 0.0, 0.0, 0.0, cloud_fallback, 0, 0.0, 0.0, 0.0, 0, 0.0)
            return (result, sigma, site_loads, n) if return_details else result

        capacities = self.service_capacity * n[active].astype(np.float64)
        raw_util = np.divide(site_loads[active], capacities, out=np.zeros_like(capacities), where=capacities > 0.0)
        util = np.minimum(raw_util, 1.0)

        mean_util = float(util.mean()) if util.size else 0.0
        std_util = float(util.std()) if util.size else 0.0
        max_util = float(util.max()) if util.size else 0.0
        overload_count = int((raw_util > 1.0).sum())

        balance_util = util[site_loads[active] > EPS] if self.load_balance_positive_only else util
        if balance_util.size == 0:
            balance = 0.0
        elif balance_util.size == 1:
            balance = 1.0
        else:
            balance_mean = float(balance_util.mean())
            balance_std = float(balance_util.std())
            balance = 1.0 - min(1.0, balance_std / (balance_mean + EPS))

        active_acus = n[active].astype(np.float64)
        total_energy = float(
            np.sum(
                self.site_idle_energy
                + active_acus * self.acu_static_energy
                + active_acus * self.acu_dynamic_energy * np.power(util, self.dynamic_energy_exponent)
            )
        )
        if self.energy_efficiency_bounds.lower() in {"milan", "vneqts"}:
            energy_min = float(self.acu_budget)
            energy_max = 2.0 * float(self.acu_budget)
            energy_eff = 1.0 - (total_energy - energy_min) / max(energy_max - energy_min, EPS)
        else:
            energy_eff = 1.0 - (total_energy - self.acu_budget) / max(self.acu_budget, EPS)
        energy_eff = float(np.clip(energy_eff, 0.0, 1.0))

        if covered_sites > 0 and covered_lambda > 0.0:
            dist_to_es = self.dist_matrix[np.arange(self.bs_num)[covered], sigma[covered]]
            avg_distance_m = float(np.average(dist_to_es, weights=self.lambda_k[covered]))
        else:
            avg_distance_m = 0.0

        w1, w2, w3, w4 = self.weights
        fitness = float(w1 * phi_site + w2 * phi_traf + w3 * balance + w4 * energy_eff)

        result = EvaluationResult(
            fitness=fitness,
            phi_site=float(phi_site),
            phi_traf=float(phi_traf),
            balance=float(balance),
            energy_eff=energy_eff,
            total_energy=total_energy,
            cloud_fallback=cloud_fallback,
            active_sites=int(active.size),
            avg_util=mean_util,
            std_util=std_util,
            max_util=max_util,
            overload_count=overload_count,
            avg_distance_m=avg_distance_m,
        )
        return (result, sigma, site_loads, n) if return_details else result

    def get_delay_list(self, deployment):
        metrics = self.evaluate_system(deployment)
        state = metrics.as_state(self.state_dim, self.acu_budget, self.bs_num, self.d_max_m)
        return state, metrics.phi_site, metrics.phi_traf, metrics.balance, metrics.energy_eff

    def get_reward(self, metrics: EvaluationResult):
        return float(metrics.fitness)

    def get_init_state(self, data_j=0):
        del data_j
        init_action = self.get_init_action()
        metrics = self.evaluate_system(init_action)
        self.last_metrics = metrics
        init_state = metrics.as_state(self.state_dim, self.acu_budget, self.bs_num, self.d_max_m)
        return init_state, metrics.phi_site, metrics.phi_traf, metrics.balance

    def next_step(self, deployment, init_delay_min=None):
        del init_delay_min
        metrics = self.evaluate_system(deployment)
        self.last_metrics = metrics
        new_state = metrics.as_state(self.state_dim, self.acu_budget, self.bs_num, self.d_max_m)
        reward = self.get_reward(metrics)
        return (
            new_state,
            metrics.phi_site,
            metrics.phi_traf,
            reward,
            metrics.balance,
            metrics.total_energy,
            metrics.cloud_fallback,
        )
