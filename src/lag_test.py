"""Step 2: model-free test for a wind-transport signature between Delhi stations (hypothesis H1).

For each ordered station pair (i -> j), compare lagged correlations of hourly log-PM2.5 changes
  c(tau) = corr( dz_i(t), dz_j(t+tau) )
on hours when the city wind blows from i towards j ("downwind" set) versus from j towards i ("upwind" set).
Transport predicts c_down(tau) > c_up(tau) for small positive lags.

Usage: python lag_test.py <prep_dir> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

prep, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
DESEASON = len(sys.argv) > 3 and sys.argv[3] == "deseason"
Z = np.load(os.path.join(prep, "panel.npz"), allow_pickle=True)
pm, lat, lon = Z["pm"], Z["lat"], Z["lon"]
times = pd.to_datetime(Z["times"].astype("datetime64[h]"))
N = pm.shape[1]
met = pd.read_parquet(os.path.join(prep, "met.parquet"))

# =====================================================================================
# 1) City-mean wind and its steadiness over the next 6 h (direction must stay constant
#    for transport between stations to be meaningful).
# =====================================================================================
# city-mean 100 m wind (vector towards which air moves), averaged over the 4 grid points
def uv(s, d):  # meteorological direction = from
    r = np.deg2rad(d); return -s * np.sin(r), -s * np.cos(r)
met["u"], met["v"] = uv(met.wind_speed_100m, met.wind_direction_100m)
W = met.groupby("time")[["u", "v"]].mean().reindex(times)
u, v = W.u.values, W.v.values
spd = np.hypot(u, v)
# steadiness over [t, t+6]: |vector mean| / scalar mean
uu = pd.Series(u).rolling(7).mean().shift(-6).values; vv = pd.Series(v).rolling(7).mean().shift(-6).values
ss = pd.Series(spd).rolling(7).mean().shift(-6).values
B = np.hypot(uu, vv) / np.maximum(ss, 1e-6)
mspd = ss  # mean speed over the window

# local coordinates (km)
lat0, lon0 = lat.mean(), lon.mean()
x = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0)); y = (lat - lat0) * 110.57
# 2) Hourly change of log PM2.5. Differencing removes slow, city-wide level changes, so the
#    remaining co-movement between stations reflects short-term propagation.
dz = np.diff(np.log(pm), axis=0, prepend=np.nan)  # hourly change of log PM2.5
if DESEASON:  # remove station-specific hour-of-day x month mean of dz (diurnal-cycle confound)
    key = times.hour.values * 100 + times.month.values
    D = pd.DataFrame(dz); D["k"] = key
    dz = (D.drop(columns="k") - D.groupby("k").transform("mean").drop(columns="k", errors="ignore")).values

# 3) Lags examined (hours) and the 'steady wind' condition used to select hours.
LAGS = np.arange(-8, 9)
wind_ang = np.arctan2(vv, uu)  # direction of mean motion over window
ok_wind = (mspd >= 1.5) & (B >= 0.8)

# Pearson correlation between station a at time t and station b at time t+lag,
# using only the selected hours; returns NaN if fewer than 150 valid pairs.
def lagcorr(a, b, idx, lag):
    j = idx + lag; m = (j >= 0) & (j < len(b)); ii, jj = idx[m], j[m]
    xa, xb = a[ii], b[jj]; f = np.isfinite(xa) & np.isfinite(xb)
    if f.sum() < 150: return np.nan, f.sum()
    return np.corrcoef(xa[f], xb[f])[0, 1], f.sum()

# =====================================================================================
# 4) For every ordered pair 3-30 km apart, compare lagged correlations on hours when the wind
#    blows from i towards j (downwind set) with hours when it blows from j towards i (upwind set).
#    A = mean over lags 1-3 h of (c_down - c_up); transport predicts A > 0.
# =====================================================================================
rows = []
for i in range(N):
    for j in range(N):
        if i == j: continue
        dx, dy = x[j] - x[i], y[j] - y[i]; d = np.hypot(dx, dy)
        if d < 3 or d > 30: continue
        bear = np.arctan2(dy, dx)
        diff = np.angle(np.exp(1j * (wind_ang - bear)))
        down = np.where(ok_wind & (np.abs(diff) <= np.deg2rad(30)))[0]
        up = np.where(ok_wind & (np.abs(np.angle(np.exp(1j * (wind_ang - bear - np.pi)))) <= np.deg2rad(30)))[0]
        if len(down) < 300 or len(up) < 300: continue
        cd = np.array([lagcorr(dz[:, i], dz[:, j], down, L)[0] for L in LAGS])
        cu = np.array([lagcorr(dz[:, i], dz[:, j], up, L)[0] for L in LAGS])
        if np.isnan(cd).any() or np.isnan(cu).any(): continue
        s_down = np.nanmean(mspd[down]) * np.cos(np.nanmean(np.abs(diff[down])))
        T = d / (s_down * 3.6)  # hours
        pos = (LAGS >= 1) & (LAGS <= 3)
        rows.append(dict(i=i, j=j, d_km=d, n_down=len(down), n_up=len(up), T_h=T,
                         A=np.mean(cd[pos]) - np.mean(cu[pos]),
                         tau_star_down=int(LAGS[np.argmax(cd)]), tau_star_up=int(LAGS[np.argmax(cu)]),
                         **{f"cd_{L}": c for L, c in zip(LAGS, cd)}, **{f"cu_{L}": c for L, c in zip(LAGS, cu)}))

# 5) Summary statistics across pairs: sign test via Wilcoxon, lag profile, and whether the
#    lag of maximum correlation grows with the advective travel time (Spearman).
R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "lag_pairs.csv"), index=False)
# unordered-pair average of A to avoid double counting
R["key"] = [tuple(sorted(k)) for k in zip(R.i, R.j)]
Au = R.groupby("key").A.mean()
w = stats.wilcoxon(Au.values)
# lead structure: is the downwind-minus-upwind curve asymmetric (positive lags > negative lags)?
cdm = R[[f"cd_{L}" for L in LAGS]].values - R[[f"cu_{L}" for L in LAGS]].values
mean_curve = cdm.mean(axis=0)
sp = stats.spearmanr(R.T_h, R.tau_star_down)
res = dict(n_ordered_pairs=len(R), n_unordered_pairs=int(len(Au)),
           median_A=float(np.median(Au)), mean_A=float(Au.mean()), frac_A_positive=float((Au > 0).mean()),
           wilcoxon_p=float(w.pvalue),
           mean_down_minus_up_by_lag={int(L): float(m) for L, m in zip(LAGS, mean_curve)},
           spearman_T_vs_tau_star=dict(rho=float(sp.statistic), p=float(sp.pvalue)),
           median_T_h=float(R.T_h.median()), median_tau_star_down=float(R.tau_star_down.median()),
           frac_tau_star_down_positive=float((R.tau_star_down > 0).mean()),
           frac_tau_star_up_positive=float((R.tau_star_up > 0).mean()),
           frac_hours_steady_wind=float(np.nanmean(ok_wind)))
json.dump(res, open(os.path.join(out, "lag_test.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].axhline(0, c="grey", lw=0.8); ax[0].plot(LAGS, mean_curve, marker="o", c="#1f5fa8")
ax[0].set_xlabel("lag tau (h)"); ax[0].set_ylabel("c_down - c_up"); ax[0].set_title("Downwind minus upwind lagged corr.")
ax[1].hist(Au.values, bins=30, color="#1f5fa8"); ax[1].axvline(0, c="k", lw=0.8)
ax[1].set_xlabel("asymmetry A (lags 1-3 h)"); ax[1].set_ylabel("station pairs"); ax[1].set_title("Per-pair asymmetry")
plt.tight_layout(); plt.savefig(os.path.join(out, "fig_lag_test.png"), dpi=200)
