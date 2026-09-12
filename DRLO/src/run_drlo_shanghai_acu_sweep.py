from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET = Path("bs_statistics_all_12files.csv")
DEFAULT_BUDGETS = [51, 91]
DEFAULT_EPISODE = 500
DEFAULT_MAX_STEP = 30
DEFAULT_DMAX_M = 3000.0
DEFAULT_SEED = 20260908
DEFAULT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DRLO Shanghai Telecom ACU-budget sensitivity experiments."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Shanghai Telecom CSV dataset.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
        help="ACU budgets to test.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=DEFAULT_EPISODE,
        help="Training episodes for each ACU budget.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=DEFAULT_MAX_STEP,
        help="DRLO environment steps per episode.",
    )
    parser.add_argument(
        "--dmax-m",
        type=float,
        default=DEFAULT_DMAX_M,
        help="Coverage radius Dmax in meters.",
    )
    parser.add_argument(
        "--service-capacity",
        type=float,
        default=None,
        help="Capacity of one ACU. Omit to use Sigma(lambda)/M, matching Shanghai VNEQTS.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=0,
        help="Increase seed by this amount after each budget.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("drlo_shanghai_acu_sweep"),
        help="Directory where all experiment outputs are saved.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def load_base_config() -> dict:
    config_path = Path("parameter.json")
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def seed_everything(seed: int) -> None:
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_config(
    base_config: dict,
    dataset: Path,
    budget: int,
    episode: int,
    max_step: int,
    dmax_m: float,
    service_capacity: float | None,
) -> dict:
    config = dict(base_config)
    config.update(
        {
            "data_path": str(dataset),
            "bs_num": 0,
            "acu_budget": int(budget),
            "n_max": 3,
            "d_max_m": float(dmax_m),
            "service_capacity": service_capacity,
            "episode": int(episode),
            "max_step": int(max_step),
            "capacity_constrained_mapping": False,
            "site_idle_energy": 0.0,
            "acu_static_energy": 1.0,
            "acu_dynamic_energy": 1.0,
            "dynamic_energy_exponent": 1.0,
            "energy_efficiency_bounds": "vneqts",
            "load_balance_positive_only": True,
            "weights": DEFAULT_WEIGHTS,
            "weight_profiles": [],
            "run_weight_profiles": [],
            "profile_name": "",
        }
    )
    config.setdefault("tanh_y", 20)
    config.setdefault("tanh_x", 0.002)
    return config


def print_required_metrics(budget: int, metrics, station_count: int) -> None:
    print(f"\n========== Shanghai DRLO ACU={budget} Result ==========")
    print(f"Fitness: {metrics.fitness:.6f}")
    print(f"Site coverage: {metrics.phi_site * 100.0:.4f}%")
    print(f"Traffic coverage: {metrics.phi_traf * 100.0:.4f}%")
    print(f"Load balance: {metrics.balance:.6f}")
    print(f"Energy efficiency: {metrics.energy_eff:.6f}")
    print(f"Total energy: {metrics.total_energy:.6f}")
    print(f"Cloud fallback BSs: {metrics.cloud_fallback} / {station_count}")
    print(f"Active edge sites: {metrics.active_sites} / {station_count}")
    print(f"Average utilization: {metrics.avg_util * 100.0:.4f}%")


def metrics_row(
    budget: int,
    seed: int,
    episode: int,
    max_step: int,
    dmax_m: float,
    effective_capacity: float,
    metrics,
    run_dir: Path,
) -> dict:
    return {
        "AcuBudget": budget,
        "Seed": seed,
        "Episode": episode,
        "MaxStep": max_step,
        "DmaxMeters": dmax_m,
        "ServiceCapacity": effective_capacity,
        "Fitness": metrics.fitness,
        "SiteCoverage": metrics.phi_site,
        "SiteCoveragePct": metrics.phi_site * 100.0,
        "TrafficCoverage": metrics.phi_traf,
        "TrafficCoveragePct": metrics.phi_traf * 100.0,
        "LoadBalance": metrics.balance,
        "EnergyEfficiency": metrics.energy_eff,
        "TotalEnergy": metrics.total_energy,
        "CloudFallbackBSs": metrics.cloud_fallback,
        "ActiveEdgeSites": metrics.active_sites,
        "AverageUtilization": metrics.avg_util,
        "AverageUtilizationPct": metrics.avg_util * 100.0,
        "RunDir": str(run_dir),
    }


def write_summary(rows: Iterable[dict], summary_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = list(rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one_budget(
    train_module,
    config: dict,
    budget: int,
    seed: int,
    station_count: int,
    total_lambda: float,
    output_dir: Path,
) -> dict:
    import tensorflow as tf

    run_dir = output_dir / f"acu_{budget:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_capacity = (
        float(config["service_capacity"])
        if config.get("service_capacity") is not None and float(config["service_capacity"]) > 0.0
        else total_lambda / max(int(budget), 1)
    )
    profile_name = f"Shanghai_ACU{budget}_ProfileA_equal"
    config["profile_name"] = profile_name

    with (run_dir / "effective_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)
    with (run_dir / "parameter.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)

    seed_everything(seed)
    tf.keras.backend.clear_session()

    original_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        print(f"\n########## Running DRLO Shanghai ACU={budget}, seed={seed} ##########")
        metrics = train_module.ddpg_play(
            config,
            profile_name=profile_name,
            profile_weights=DEFAULT_WEIGHTS,
        )
    finally:
        os.chdir(original_cwd)

    print_required_metrics(budget, metrics, station_count)
    return metrics_row(
        budget=budget,
        seed=seed,
        episode=int(config["episode"]),
        max_step=int(config["max_step"]),
        dmax_m=float(config["d_max_m"]),
        effective_capacity=effective_capacity,
        metrics=metrics,
        run_dir=run_dir,
    )


def main() -> int:
    args = parse_args()
    dataset = resolve_path(args.dataset)
    output_root = resolve_path(args.output_root)
    output_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset.exists():
        raise FileNotFoundError(f"Shanghai dataset not found: {dataset}")

    import train
    import utils

    base_config = load_base_config()
    bs_df = utils.load_base_stations(str(dataset), 0)
    station_count = len(bs_df)
    total_lambda = float(bs_df["lambda_k"].sum())
    shutil.copy2(dataset, output_dir / dataset.name)

    rows = []
    for index, budget in enumerate(args.budgets):
        seed = args.seed + index * args.seed_step
        config = make_config(
            base_config=base_config,
            dataset=dataset,
            budget=budget,
            episode=args.episode,
            max_step=args.max_step,
            dmax_m=args.dmax_m,
            service_capacity=args.service_capacity,
        )
        row = run_one_budget(
            train_module=train,
            config=config,
            budget=budget,
            seed=seed,
            station_count=station_count,
            total_lambda=total_lambda,
            output_dir=output_dir,
        )
        rows.append(row)

    summary_path = output_dir / "drlo_shanghai_acu_sweep_summary.csv"
    write_summary(rows, summary_path)
    print(f"\nSaved DRLO Shanghai ACU sweep summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
