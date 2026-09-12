"""Convert two months of Milan telecom grid CSV files into VNEQTS workloads.

Each Milan grid cell is treated as one candidate edge site.  Coordinates are
the polygon centroid from milano-grid.geojson.  The default workload is the
mean Internet activity per observed 10-minute slot; this is an activity index,
not a session arrival rate.  By default, all daily files from November and
December 2013 are processed.  A normalized five-metric composite is available
for sensitivity experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = ("sms-in", "sms-out", "call-in", "call-out", "internet")
N_CELLS = 10_000
INPUT_MONTH_PATTERNS = ("2013-11-*.csv", "2013-12-*.csv")


def polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    twice_area = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(twice_area) < 1e-15:
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))
    return cx / (3.0 * twice_area), cy / (3.0 * twice_area)


def read_centroids(path: Path) -> dict[int, tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, tuple[float, float]] = {}
    for feature in data["features"]:
        cell_id = int(feature["properties"]["cellId"])
        geometry = feature["geometry"]
        if geometry["type"] != "Polygon":
            raise ValueError(f"cell {cell_id}: expected Polygon geometry")
        lon, lat = polygon_centroid(geometry["coordinates"][0])
        result[cell_id] = lat, lon
    if set(result) != set(range(1, N_CELLS + 1)):
        raise ValueError("GeoJSON must contain cellId 1..10000 exactly once")
    return result


def aggregate(files: list[Path]):
    sums = [[0.0] * len(METRICS) for _ in range(N_CELLS)]
    observed = [[0] * len(METRICS) for _ in range(N_CELLS)]
    rows = 0
    timestamps: set[str] = set()
    file_quality = []
    for path in files:
        file_rows = 0
        file_timestamps: dict[str, int] = {}
        seen = bytearray(144 * N_CELLS)
        duplicate_keys = 0
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != ["datetime", "squareid", *METRICS]:
                raise ValueError(f"unexpected header in {path}")
            for row in reader:
                cell0 = int(row["squareid"]) - 1
                if not 0 <= cell0 < N_CELLS:
                    raise ValueError(f"invalid squareid in {path}: {cell0 + 1}")
                timestamp = row["datetime"]
                timestamps.add(timestamp)
                if timestamp not in file_timestamps:
                    if len(file_timestamps) >= 144:
                        raise ValueError(f"more than 144 timestamps in {path}")
                    file_timestamps[timestamp] = len(file_timestamps)
                key = file_timestamps[timestamp] * N_CELLS + cell0
                if seen[key]:
                    duplicate_keys += 1
                seen[key] = 1
                for j, name in enumerate(METRICS):
                    text = row[name]
                    if text != "":
                        value = float(text)
                        if not math.isfinite(value) or value < 0.0:
                            raise ValueError(f"invalid {name} in {path}")
                        sums[cell0][j] += value
                        observed[cell0][j] += 1
                rows += 1
                file_rows += 1
        unique_keys = sum(seen)
        if duplicate_keys:
            raise ValueError(f"{path} contains {duplicate_keys} duplicate keys")
        file_quality.append({
            "file": path.name,
            "rows": file_rows,
            "timestamps": len(file_timestamps),
            "unique_keys": unique_keys,
            "missing_grid_time_keys": 144 * N_CELLS - unique_keys,
        })
        print(f"processed {path.name}", flush=True)
    return sums, observed, rows, len(timestamps), file_quality


def workloads(sums, observed, mode: str, weights: list[float]):
    means = [
        [sums[i][j] / observed[i][j] if observed[i][j] else 0.0
         for j in range(len(METRICS))]
        for i in range(N_CELLS)
    ]
    if mode in METRICS:
        column = METRICS.index(mode)
        return [row[column] for row in means], means

    # Normalize each metric by its citywide mean before weighting.  This keeps
    # Internet's larger numerical scale from silently dominating the composite.
    city_means = [sum(row[j] for row in means) / N_CELLS for j in range(len(METRICS))]
    result = []
    for row in means:
        result.append(sum(weights[j] * row[j] / city_means[j]
                          for j in range(len(METRICS)) if city_means[j] > 0.0))
    return result, means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--geojson", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metric", choices=[*METRICS, "composite"], default="internet")
    parser.add_argument("--block-size", type=int, default=2,
                        help="aggregate BxB source cells (default 2 gives 2500 sites; use 1 for all 10000)")
    parser.add_argument("--weights", default="0.2,0.2,0.2,0.2,0.2",
                        help="five comma-separated weights for composite mode")
    args = parser.parse_args()

    geojson = args.geojson or args.input_dir / "milano-grid.geojson"
    output = args.output or args.input_dir / "milan_bs_workload.csv"
    files = sorted(
        path
        for pattern in INPUT_MONTH_PATTERNS
        for path in args.input_dir.glob(pattern)
    )
    if not files:
        patterns = ", ".join(INPUT_MONTH_PATTERNS)
        raise SystemExit(f"no Milan daily CSV files found matching: {patterns}")
    weights = [float(x) for x in args.weights.split(",")]
    if args.block_size <= 0 or 100 % args.block_size != 0:
        raise SystemExit("block-size must be a positive divisor of 100")
    if len(weights) != 5 or any(x < 0 for x in weights) or sum(weights) <= 0:
        raise SystemExit("weights must be five non-negative values with positive sum")
    weight_sum = sum(weights)
    weights = [x / weight_sum for x in weights]

    centroids = read_centroids(geojson)
    sums, observed, row_count, time_slots, file_quality = aggregate(files)
    load, means = workloads(sums, observed, args.metric, weights)

    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(["bs_id", "latitude", "longitude", "workload",
                         "observed_slots", *[f"mean_{x}" for x in METRICS]])
        metric_index = METRICS.index(args.metric) if args.metric in METRICS else None
        block_id = 0
        width = 100 // args.block_size
        for block_row in range(width):
          for block_col in range(width):
            block_id += 1
            cells = [
                (block_row * args.block_size + dr) * 100
                + block_col * args.block_size + dc + 1
                for dr in range(args.block_size) for dc in range(args.block_size)
            ]
            lat = sum(centroids[c][0] for c in cells) / len(cells)
            lon = sum(centroids[c][1] for c in cells) / len(cells)
            block_load = sum(load[c - 1] for c in cells)
            slots = min(
                observed[c - 1][metric_index]
                if metric_index is not None else max(observed[c - 1])
                for c in cells
            )
            block_means = [sum(means[c - 1][j] for c in cells) for j in range(len(METRICS))]
            writer.writerow([
                f"MI_B{block_id}", f"{lat:.8f}", f"{lon:.8f}",
                f"{max(block_load, 1e-6):.10f}", slots,
                *[f"{x:.10f}" for x in block_means],
            ])

    metadata = {
        "input_files": [p.name for p in files],
        "input_rows": row_count,
        "distinct_10min_slots": time_slots,
        "per_file_quality": file_quality,
        "source_grid_cells": N_CELLS,
        "spatial_block_size": args.block_size,
        "output_candidate_sites": (100 // args.block_size) ** 2,
        "workload_definition": (
            f"mean {args.metric} activity per nonblank 10-minute slot"
            if args.metric in METRICS else
            "weighted sum of per-metric cell means normalized by citywide metric means"
        ),
        "metric": args.metric,
        "composite_weights": dict(zip(METRICS, weights)) if args.metric == "composite" else None,
        "spatial_aggregation": "workloads summed within each BxB block; coordinates are mean source-cell centroids",
        "missing_policy": "blank values excluded from cell metric mean; block workload floored at 1e-6",
        "timezone": "CSV datetime is UTC; file dates correspond to Europe/Rome local dates",
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output} and metadata", flush=True)


if __name__ == "__main__":
    main()
