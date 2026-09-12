import math

import numpy as np
import pandas as pd


T_OBS_SECONDS = 183.0 * 24.0 * 3600.0


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _column(df, *names, fallback_idx=None):
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        found = normalized.get(name.strip().lower())
        if found is not None:
            return found
    if fallback_idx is not None:
        return df.columns[fallback_idx]
    raise KeyError(f"Missing required column. Tried: {names}")


def init_sys(csv_file_path, num_servers=71, **kwargs):
    df = pd.read_csv(csv_file_path)

    bs_id_col = _column(df, "bs_id", "基站ID", fallback_idx=0)
    lat_col = _column(df, "latitude", "纬度", fallback_idx=1)
    lon_col = _column(df, "longitude", "经度", fallback_idx=2)

    lats = df[lat_col].values.astype(float)
    lons = df[lon_col].values.astype(float)

    try:
        lambda_col = _column(df, "lambda")
        lambdas = df[lambda_col].values.astype(float)
    except KeyError:
        duration_col = _column(df, "total_duration(s)", "总停留时长(秒)", fallback_idx=3)
        lambdas = df[duration_col].values.astype(float) / T_OBS_SECONDS

    lambdas = np.where(lambdas < 1e-6, 1e-6, lambdas)

    num_bs = len(df)
    num_srv = num_servers
    acu_capacity = float(np.sum(lambdas) / num_srv)
    all_mus = np.full(num_bs, acu_capacity, dtype=float)

    dists = np.zeros((num_bs, num_bs), dtype=float)
    for i in range(num_bs):
        for j in range(i + 1, num_bs):
            dist = calculate_haversine_distance(lats[i], lons[i], lats[j], lons[j])
            dists[i, j] = dists[j, i] = dist

    covers = None
    d0 = None
    return num_bs, num_srv, num_bs, lambdas, all_mus, dists, covers, d0


if __name__ == "__main__":
    S, E, U, lambdas, mus, dists, covers, d0 = init_sys(
        csv_file_path="bs_statistics_all_12files.csv",
        num_servers=71,
    )
    print("First lambda:", lambdas[0])
    print("ACU capacity C0:", mus[0])
    print("Distance[0,1] km:", dists[0][1])
