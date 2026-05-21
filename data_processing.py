import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

INPUT_PATTERN = "oisst-avhrr-v02r01.202512*.nc"

FULL_OUT_DIR = "oisst_unpacked_full_csv"
WEEKLY_OUT_DIR = "oisst_processed_weekly_westcoast_csv"

LAT_MIN, LAT_MAX = 20, 50
LON_MIN, LON_MAX = 230, 250

N_SAMPLE_NODES = 200
RANDOM_SEED = 42

os.makedirs(FULL_OUT_DIR, exist_ok=True)
os.makedirs(WEEKLY_OUT_DIR, exist_ok=True)


def extract_date_from_filename(path):
    return os.path.basename(path).split(".")[-2]


def load_oisst(path):
    ds = xr.open_dataset(path)

    lat = ds["lat"].values
    lon = ds["lon"].values

    sst = ds["sst"].squeeze().values
    anom = ds["anom"].squeeze().values if "anom" in ds else None
    err = ds["err"].squeeze().values if "err" in ds else None
    ice = ds["ice"].squeeze().values if "ice" in ds else None

    ds.close()
    return lat, lon, sst, anom, err, ice


def grid_to_long_csv(date, lat, lon, sst, anom=None, err=None, ice=None):
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    df = pd.DataFrame({
        "date": date,
        "lat": lat_grid.ravel(),
        "lon": lon_grid.ravel(),
        "sst": sst.ravel(),
    })

    if anom is not None:
        df["anom"] = anom.ravel()
    if err is not None:
        df["err"] = err.ravel()
    if ice is not None:
        df["ice"] = ice.ravel()

    return df


def crop_region(lat, lon, arr):
    lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
    return lat[lat_mask], lon[lon_mask], arr[np.ix_(lat_mask, lon_mask)]


files = sorted(glob.glob(INPUT_PATTERN))

if len(files) == 0:
    raise FileNotFoundError("No .nc files found. Put this script next to your OISST files.")

print(f"Found {len(files)} files.")


#Full daily csv files (no cropping)

print("\nSaving full daily CSV files...")

for path in files:
    date = extract_date_from_filename(path)
    lat, lon, sst, anom, err, ice = load_oisst(path)

    df_full = grid_to_long_csv(date, lat, lon, sst, anom, err, ice)

    out_path = os.path.join(FULL_OUT_DIR, f"full_{date}.csv")
    df_full.to_csv(out_path, index=False)

    print(f"Saved {out_path}")

print("\nFinished full daily CSV unpacking.")


#Cropped west coast daily arrays

dates = []
sst_crops = []

lat_crop_ref = None
lon_crop_ref = None

print("\nLoading cropped West Coast region...")

for path in files:
    date = extract_date_from_filename(path)
    lat, lon, sst, anom, err, ice = load_oisst(path)

    lat_crop, lon_crop, sst_crop = crop_region(lat, lon, sst)

    if lat_crop_ref is None:
        lat_crop_ref = lat_crop
        lon_crop_ref = lon_crop

    dates.append(date)
    sst_crops.append(sst_crop)

dates = np.array(dates)
sst_crops = np.stack(sst_crops, axis=0)

print("Cropped daily SST shape:", sst_crops.shape)


#Weekly cropped csv files

week_slices = [
    ("week1_20251201_20251207", 0, 7),
    ("week2_20251208_20251214", 7, 14),
    ("week3_20251215_20251221", 14, 21),
    ("week4_20251222_20251231", 21, 31),
]

weekly_sst = []

print("\nSaving weekly cropped CSV files...")

for week_name, start, end in week_slices:
    if start >= len(sst_crops):
        continue

    end = min(end, len(sst_crops))
    week_avg = np.nanmean(sst_crops[start:end], axis=0)
    weekly_sst.append(week_avg)

    df_week = grid_to_long_csv(
        date=week_name,
        lat=lat_crop_ref,
        lon=lon_crop_ref,
        sst=week_avg,
    )

    df_week = df_week.rename(columns={"date": "week", "sst": "sst_weekly_avg"})

    out_path = os.path.join(WEEKLY_OUT_DIR, f"{week_name}_grid.csv")
    df_week.to_csv(out_path, index=False)

    print(f"Saved {out_path}")

weekly_sst = np.stack(weekly_sst, axis=0)


#Sample valid ocean nodes

print("\nSampling valid ocean nodes...")

valid_mask = np.all(np.isfinite(weekly_sst), axis=0)
valid_indices = np.argwhere(valid_mask)

if len(valid_indices) < N_SAMPLE_NODES:
    raise ValueError(f"Only {len(valid_indices)} valid nodes found.")

rng = np.random.default_rng(RANDOM_SEED)
chosen = valid_indices[rng.choice(len(valid_indices), size=N_SAMPLE_NODES, replace=False)]

rows = []

for node_id, (i, j) in enumerate(chosen):
    row = {
        "node_id": node_id,
        "lat_index": int(i),
        "lon_index": int(j),
        "lat": float(lat_crop_ref[i]),
        "lon": float(lon_crop_ref[j]),
    }

    for w in range(weekly_sst.shape[0]):
        row[f"sst_week{w+1}"] = float(weekly_sst[w, i, j])

    rows.append(row)

nodes_df = pd.DataFrame(rows)

nodes_path = os.path.join(WEEKLY_OUT_DIR, "sampled_nodes_for_experiments.csv")
nodes_df.to_csv(nodes_path, index=False)

print(f"Saved {nodes_path}")


#Experiment-ready long csv

experiment_rows = []

for _, row in nodes_df.iterrows():
    for w in range(weekly_sst.shape[0]):
        experiment_rows.append({
            "node_id": int(row["node_id"]),
            "lat": row["lat"],
            "lon": row["lon"],
            "week": w + 1,
            "sst": row[f"sst_week{w+1}"],
        })

experiment_df = pd.DataFrame(experiment_rows)

experiment_path = os.path.join(WEEKLY_OUT_DIR, "experiment_ready_sampled_sst.csv")
experiment_df.to_csv(experiment_path, index=False)

print(f"Saved {experiment_path}")

print("\nDONE.")
print("Best file for experiments:")
print(experiment_path)