import csv
import json

import numpy as np

import train
import utils


MILAN_DATA_PATH = "milan_bs_workload.csv"
DATASET_PREFIX = "Milan"
MILAN_VNEQTS_SETTINGS = {
    "acu_budget": 71,
    "n_max": 3,
    "d_max_m": 1500.0,
    "service_capacity": 11000.0,
    "episode": 500,
    "capacity_constrained_mapping": False,
    "site_idle_energy": 0.0,
    "acu_static_energy": 1.0,
    "acu_dynamic_energy": 1.0,
    "dynamic_energy_exponent": 1.0,
    "energy_efficiency_bounds": "vneqts",
    "load_balance_positive_only": True,
    "weights": [0.25, 0.25, 0.25, 0.25],
    "weight_profiles": [
        {
            "name": "ProfileA_equal",
            "weights": [0.25, 0.25, 0.25, 0.25],
        }
    ],
    "run_weight_profiles": ["ProfileA_equal"],
}


def _selected_profiles(config):
    profiles = train._iter_weight_profiles(config)
    if profiles:
        return profiles
    return [(config.get("profile_name", "ProfileA_equal"), config.get("weights", [0.25, 0.25, 0.25, 0.25]))]


def _print_required_metrics(profile_name, metrics, station_count):
    print(f"\n========== {DATASET_PREFIX} DRLO Result ({profile_name}) ==========")
    print(f"Fitness: {metrics.fitness:.6f}")
    print(f"Site coverage: {metrics.phi_site * 100.0:.4f}%")
    print(f"Traffic coverage: {metrics.phi_traf * 100.0:.4f}%")
    print(f"Load balance: {metrics.balance:.6f}")
    print(f"Energy efficiency: {metrics.energy_eff:.6f}")
    print(f"Total energy: {metrics.total_energy:.6f}")
    print(f"Cloud fallback BSs: {metrics.cloud_fallback} / {station_count}")
    print(f"Active edge sites: {metrics.active_sites} / {station_count}")
    print(f"Average utilization: {metrics.avg_util * 100.0:.4f}%")


def main():
    with open("parameter.json", encoding="utf-8") as f:
        base_config = json.load(f)

    config = dict(base_config)
    config["data_path"] = MILAN_DATA_PATH
    config["bs_num"] = 0
    config.update(MILAN_VNEQTS_SETTINGS)
    station_count = utils.resolve_bs_num(MILAN_DATA_PATH, 0)

    print(
        "Milan VNEQTS-matched settings: "
        f"N={station_count}, M={config['acu_budget']}, n_max={config['n_max']}, "
        f"Dmax={config['d_max_m']}m, C={config['service_capacity']}, "
        f"mapping={'capacity-constrained' if config['capacity_constrained_mapping'] else 'nearest-within-radius'}, "
        f"energy=n_i*({config['acu_static_energy']} + "
        f"{config['acu_dynamic_energy']}*u_i^{config['dynamic_energy_exponent']}), "
        f"weights={config['weights']}"
    )

    results = []
    for profile_name, weights in _selected_profiles(config):
        milan_profile_name = f"{DATASET_PREFIX}_{profile_name}"
        profile_config = dict(config)
        profile_config["weights"] = weights
        profile_config["profile_name"] = milan_profile_name
        print(f"\n########## Running {milan_profile_name}: weights={weights} ##########")
        metrics = train.ddpg_play(
            profile_config,
            profile_name=milan_profile_name,
            profile_weights=weights,
        )
        _print_required_metrics(profile_name, metrics, station_count)
        results.append((profile_name, weights, metrics))

    summary_filename = "weight_profile_summary_DRLO_Milan.csv"
    with open(summary_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Profile",
                "w_site",
                "w_traffic",
                "w_balance",
                "w_energy",
                "Fitness",
                "SiteCoverage",
                "TrafficCoverage",
                "LoadBalance",
                "EnergyEfficiency",
                "TotalEnergy",
                "CloudFallbackBSs",
                "ActiveEdgeSites",
                "AverageUtilization",
            ]
        )
        for profile_name, weights, metrics in results:
            w = np.asarray(weights, dtype=np.float64).ravel()
            writer.writerow(
                [
                    profile_name,
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                    metrics.fitness,
                    metrics.phi_site,
                    metrics.phi_traf,
                    metrics.balance,
                    metrics.energy_eff,
                    metrics.total_energy,
                    metrics.cloud_fallback,
                    metrics.active_sites,
                    metrics.avg_util,
                ]
            )
    print(f"\nSaved Milan profile summary: {summary_filename}")


if __name__ == "__main__":
    main()
