import numpy as np


D_MAX_KM = 4.8
M_BUDGET = 71
N_MAX = 3
DEFAULT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
B_EPS = 1e-9


class fitness:

    def __init__(self, all_mus, req_rate, dists, cloud_delay=0.05,
                 weights=DEFAULT_WEIGHTS, service_distance_km=D_MAX_KM,
                 acu_budget=M_BUDGET, exclude_zero_load_from_balance=False):
        if len(weights) != 4:
            raise ValueError("weights must contain four values")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("fitness weights must sum to 1.0")

        self.all_mus = all_mus
        self.req_rate = req_rate.astype(float)
        self.dists = dists
        self.num_srv = M_BUDGET
        self.num_bs = len(req_rate)
        self.total_lambda = float(np.sum(self.req_rate))
        self.c0 = float(all_mus[0]) if len(all_mus) else 0.0

        self.station_order = np.argsort(-self.req_rate, kind="stable")
        self.cloud_delay = cloud_delay
        self.weights = tuple(float(w) for w in weights)
        self.service_distance_km = float(service_distance_km)
        self.acu_budget = int(acu_budget)
        self.exclude_zero_load_from_balance = bool(
            exclude_zero_load_from_balance
        )
        if self.service_distance_km <= 0:
            raise ValueError("service_distance_km must be positive")
        if self.acu_budget <= 0:
            raise ValueError("acu_budget must be positive")

    def solution_to_counts(self, solution):
        counts = np.zeros(self.num_bs, dtype=int)
        overflow = 0
        for raw_idx in solution:
            idx = int(raw_idx) % self.num_bs
            if counts[idx] < N_MAX:
                counts[idx] += 1
            else:
                overflow += 1

        if overflow:
            order = np.argsort(-self.req_rate)
            for idx in order:
                while overflow and counts[idx] < N_MAX:
                    counts[idx] += 1
                    overflow -= 1
                if overflow == 0:
                    break
        return counts

    def is_feasible_solution(self, solution):
        candidate = np.asarray(solution)
        if candidate.ndim != 1 or len(candidate) != self.acu_budget:
            return False
        if not np.issubdtype(candidate.dtype, np.integer):
            if (not np.all(np.isfinite(candidate))
                    or not np.all(candidate == np.floor(candidate))):
                return False
        candidate = candidate.astype(int, copy=False)
        if np.any(candidate < 0) or np.any(candidate >= self.num_bs):
            return False
        counts = np.bincount(candidate, minlength=self.num_bs)
        return bool(np.max(counts, initial=0) <= N_MAX)

    def repair_randomly(self, solution):
        """Relocate only ACUs exceeding the per-site deployment limit."""
        repaired = np.asarray(solution, dtype=int).copy()
        if repaired.ndim != 1 or len(repaired) != self.acu_budget:
            raise ValueError("Each solution must encode exactly the ACU budget")
        if self.acu_budget > self.num_bs * N_MAX:
            raise ValueError("The ACU budget exceeds available deployment capacity")
        if np.any(repaired < 0) or np.any(repaired >= self.num_bs):
            raise ValueError("A gene refers to an invalid base-station index")

        counts = np.zeros(self.num_bs, dtype=int)
        overflow_positions = []
        for position, site in enumerate(repaired):
            if counts[site] < N_MAX:
                counts[site] += 1
            else:
                overflow_positions.append(position)
        for position in overflow_positions:
            available = np.flatnonzero(counts < N_MAX)
            site = int(np.random.choice(available))
            repaired[position] = site
            counts[site] += 1
        return repaired

    def random_solution(self):
        counts = []
        remaining = self.acu_budget
        while remaining > 0:
            count = np.random.randint(1, min(N_MAX, remaining) + 1)
            counts.append(count)
            remaining -= count

        active_sites = np.random.choice(
            self.num_bs, size=len(counts), replace=False
        )
        counts = np.asarray(counts, dtype=int)
        np.random.shuffle(counts)
        solution = np.repeat(active_sites, counts)
        np.random.shuffle(solution)
        return solution

    def random_population(self, population_size):
        return np.vstack([
            self.random_solution() for _ in range(population_size)
        ])

    def _compute_metrics(self, solution):
        counts = self.solution_to_counts(solution)
        active_sites = np.flatnonzero(counts > 0)

        if len(active_sites) == 0:
            return self._empty_metrics(counts)

        active_counts = counts[active_sites].astype(float)
        capacities = self.c0 * active_counts
        remaining_capacity = capacities.copy()
        sigma = np.full(self.num_bs, -1, dtype=int)
        assigned_dist = np.full(self.num_bs, np.inf, dtype=float)
        site_loads = np.zeros(self.num_bs, dtype=float)

        sub_dists = self.dists[:, active_sites]
        nearest_active_pos = np.argmin(sub_dists, axis=1)
        for station in self.station_order:
            demand = self.req_rate[station]
            active_pos = nearest_active_pos[station]
            distance = sub_dists[station, active_pos]
            if (distance <= self.service_distance_km
                    and demand <= remaining_capacity[active_pos] + B_EPS):
                site = active_sites[active_pos]
                sigma[station] = site
                assigned_dist[station] = distance
                site_loads[site] += demand
                remaining_capacity[active_pos] -= demand

        covered_mask = sigma >= 0
        covered_sites = int(np.sum(covered_mask))
        cloud_fallback = self.num_bs - covered_sites
        covered_lambda = float(np.sum(self.req_rate[covered_mask]))

        active_loads = site_loads[active_sites]
        rho = np.divide(active_loads, capacities, out=np.zeros_like(active_loads), where=capacities > 0)
        if np.any(active_loads > capacities + B_EPS):
            raise RuntimeError("Hard ES capacity constraint was violated")
        rho_clipped = np.clip(rho, 0.0, 1.0)

        phi_site = covered_sites / self.num_bs
        phi_traf = covered_lambda / self.total_lambda if self.total_lambda > 0 else 0.0

        balance_utilizations = rho_clipped
        if self.exclude_zero_load_from_balance:
            balance_utilizations = rho_clipped[active_loads > B_EPS]

        avg_utilization = (
            float(np.mean(balance_utilizations))
            if len(balance_utilizations) > 0 else 0.0
        )

        if len(balance_utilizations) >= 2:
            mean_rho = avg_utilization
            cv = float(np.std(balance_utilizations) / (mean_rho + B_EPS))
            balance = 1.0 - min(1.0, cv)
        elif len(balance_utilizations) == 1:
            balance = 1.0
        else:
            balance = 0.0

        total_energy = float(np.sum(active_counts * (1.0 + rho_clipped)))
        energy_eff = 1.0 - (total_energy - self.acu_budget) / self.acu_budget
        energy_eff = max(0.0, min(1.0, energy_eff))

        w_site, w_traf, w_balance, w_energy = self.weights
        F = (w_site * phi_site + w_traf * phi_traf
             + w_balance * balance + w_energy * energy_eff)

        return {
            "F": F,
            "phi_site": phi_site,
            "phi_traf": phi_traf,
            "balance": balance,
            "energy_eff": energy_eff,
            "total_energy": total_energy,
            "avg_utilization": avg_utilization,
            "cloud_fallback": cloud_fallback,
            "active_sites": active_sites,
            "counts": counts,
            "sigma": sigma,
            "site_loads": site_loads,
            "rho": rho_clipped,
            "capacities": capacities,
            "nearest_dist": assigned_dist,
        }

    def _empty_metrics(self, counts):
        return {
            "F": 0.0,
            "phi_site": 0.0,
            "phi_traf": 0.0,
            "balance": 0.0,
            "energy_eff": 0.0,
            "total_energy": 0.0,
            "avg_utilization": 0.0,
            "cloud_fallback": self.num_bs,
            "active_sites": np.array([], dtype=int),
            "counts": counts,
            "sigma": np.full(self.num_bs, -1, dtype=int),
            "site_loads": np.zeros(self.num_bs, dtype=float),
            "rho": np.array([], dtype=float),
            "capacities": np.array([], dtype=float),
            "nearest_dist": np.full(self.num_bs, np.inf, dtype=float),
        }

    def fn(self, solution):
        m = self._compute_metrics(solution)
        return max(1.0 - m["F"], 1e-9)

    def fn_without_repair(self, solution):
        """Return zero fitness for an infeasible genotype without repairing it."""
        if not self.is_feasible_solution(solution):
            return 1.0
        return self.fn(solution)

    def unified_evaluate(self, solution):
        m = self._compute_metrics(solution)
        return {
            "F": round(m["F"], 6),
            "phi_site_pct": round(m["phi_site"] * 100, 4),
            "phi_traf_pct": round(m["phi_traf"] * 100, 4),
            "balance": round(m["balance"], 6),
            "energy_eff": round(m["energy_eff"], 6),
            "total_energy": round(m["total_energy"], 6),
            "avg_utilization_pct": round(m["avg_utilization"] * 100, 4),
            "cloud_fallback": int(m["cloud_fallback"]),
            "active_site_count": int(len(m["active_sites"])),
            "acu_count": int(np.sum(m["counts"])),
            "c0": round(self.c0, 6),
        }
