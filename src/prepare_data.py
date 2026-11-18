"""Step 1: build the hourly Delhi PM2.5 panel and ERA5 meteorology.

Usage: python prepare_data.py <raw_parquet_dir> <coords_csv> <out_dir> [end_year=2025]

Outputs (out_dir):
  panel.npz      pm[T,N] hourly PM2.5 (NaN = missing), times (IST, naive), station ids/names, lat, lon
  met.parquet    hourly ERA5 (via Open-Meteo, IST) per ERA5 grid point + station -> grid map
  stations.csv   selected stations with completeness
Notes:
  * Archive timestamps are labelled UTC but are local IST (verified from diurnal cycle); we treat them as IST.
  * Quarter-hour values are near-duplicates, so an hour is valid if >=1 quarter-hour value is valid.
"""
import sys, os, glob, json, time
import numpy as np, pandas as pd, requests

raw_dir, coords_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
END = int(sys.argv[4]) if len(sys.argv) > 4 else 2025
os.makedirs(out, exist_ok=True)

# =====================================================================================
# 1) Hourly PM2.5 panel with quality control
#    - timestamps in the archive are labelled UTC but are local IST, so we keep the wall clock
#    - an hour is valid if at least one quarter-hour value is valid (quarter-hours are near-duplicates)
# =====================================================================================
# ---------- PM2.5 ----------
df = pd.concat([pd.read_parquet(f, columns=["Station ID", "Station Name", "Timestamp", "PM2.5", "PM10"])
                for f in sorted(glob.glob(os.path.join(raw_dir, "delhi_20*.parquet")))], ignore_index=True)
df["t"] = df["Timestamp"].dt.tz_localize(None).dt.floor("h")          # IST wall clock
pm = df["PM2.5"].where((df["PM2.5"] >= 2) & (df["PM2.5"] < 999))       # QC: detection limit, cap
H = df.assign(pm=pm).groupby(["Station ID", "t"]).agg(pm=("pm", "mean"), pm10=("PM10", "mean")).reset_index()
H.loc[(H.pm10.notna()) & (H.pm > H.pm10 * 1.05), "pm"] = np.nan       # QC: PM2.5 > PM10
names = df.groupby("Station ID")["Station Name"].first()

times = pd.date_range("2019-01-01 00:00", f"{END}-12-31 23:00", freq="h")
P = H.pivot(index="t", columns="Station ID", values="pm").reindex(times)

# flat-line QC: runs of >=6 identical hourly values -> missing (sensor stuck)
for c in P.columns:
    s = P[c]
    grp = (s != s.shift()).cumsum()
    run = s.groupby(grp).transform("size")
    P.loc[(run >= 6) & s.notna(), c] = np.nan

# 2) Station selection: complete enough in every training year and usable in later years.
# station selection: >=75% completeness in each year 2019-2022, and >=50% in every later year up to END
comp = P.notna().groupby(P.index.year).mean().T
keep = comp.index[(comp[[2019, 2020, 2021, 2022]] >= 0.75).all(axis=1) & (comp[list(range(2023, END + 1))] >= 0.50).all(axis=1)]
coords = pd.read_csv(coords_csv).set_index("station_id")
keep = [k for k in keep if k in coords.index]
P = P[keep]
lat = coords.loc[keep, "lat"].values.astype(float); lon = coords.loc[keep, "lon"].values.astype(float)
st = pd.DataFrame({"station_id": keep, "name": names.loc[keep].values, "lat": lat, "lon": lon})
st = st.join(comp.loc[keep].add_prefix("comp_").reset_index(drop=True))
st.to_csv(os.path.join(out, "stations.csv"), index=False)
print(f"stations kept: {len(keep)} of {comp.shape[0]}")
print("overall completeness of kept panel:", round(P.notna().values.mean(), 3))

# =====================================================================================
# 3) Meteorology: ERA5 at the 0.25 deg grid points nearest to the stations
#    (10 m and 100 m wind, boundary-layer height), requested directly in IST.
# =====================================================================================
# ---------- ERA5 via Open-Meteo (archive API, models=era5, IST) ----------
grid = sorted({(round(a * 4) / 4, round(b * 4) / 4) for a, b in zip(lat, lon)})
st_grid = [grid.index((round(a * 4) / 4, round(b * 4) / 4)) for a, b in zip(lat, lon)]
vars_ = "wind_speed_10m,wind_direction_10m,wind_speed_100m,wind_direction_100m,boundary_layer_height"
mets = []
for gi, (a, b) in enumerate(grid):
    for attempt in range(5):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
            latitude=a, longitude=b, start_date="2019-01-01", end_date=f"{END}-12-31", hourly=vars_,
            wind_speed_unit="ms", timezone="Asia/Kolkata", models="era5"), timeout=120)
        if r.ok: break
        time.sleep(10)
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"]); h["time"] = pd.to_datetime(h["time"]); h["grid"] = gi
    h["grid_lat"], h["grid_lon"] = a, b
    mets.append(h)
    print("grid", gi, (a, b), "rows", len(h), "stations", st_grid.count(gi))
met = pd.concat(mets, ignore_index=True)
met.to_parquet(os.path.join(out, "met.parquet"), index=False)

# 4) Save the panel (time x station matrix) plus station metadata for all later steps.
np.savez_compressed(os.path.join(out, "panel.npz"), pm=P.values.astype(np.float32),
                    times=times.values.astype("datetime64[h]").astype(np.int64), station_ids=np.array(keep),
                    names=names.loc[keep].values.astype(str), lat=lat, lon=lon, st_grid=np.array(st_grid))
json.dump({"n_stations": len(keep), "grid": grid, "T": len(times)}, open(os.path.join(out, "prep_meta.json"), "w"), indent=1)
print("done")
