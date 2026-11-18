"""Step 4: statistics and regime analysis on test-set (2024) predictions.

Usage: python evaluate_stats.py <predictions.parquet> <met_origin.parquet> <panel.npz> <out_dir> <tag>
Works for both the XGBoost and the GNN prediction files (columns t, j, h, y, <arms...>).
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

pred_f, met_f, panel_f, out, tag = sys.argv[1:6]; os.makedirs(out, exist_ok=True)
P = pd.read_parquet(pred_f); met = pd.read_parquet(met_f).set_index("t")
Z = np.load(panel_f, allow_pickle=True); lat, lon = Z["lat"], Z["lon"]
times = pd.to_datetime(Z["times"].astype("datetime64[h]"))
arms = [c for c in P.columns if c.startswith("G")]
base_arm = "G0_none"
others = [c for c in ["persistence", "seasonal_naive"] if c in P.columns]
rng = np.random.default_rng(0)

# =====================================================================================
# Regime labels for every forecast (used for evaluation only, never as model inputs):
# season of the target hour, ventilation tercile at the origin, transport-ratio bin.
# =====================================================================================
# ---------- regimes ----------
tt = P.t.values; tgt = tt + P.h.values
ttime = times[np.minimum(tgt, len(times) - 1)]
md = ttime.month * 100 + ttime.day
season = np.select([(md >= 1016) & (md <= 1130), (md >= 1216) | (md <= 115), (ttime.month >= 4) & (ttime.month <= 6),
                    (ttime.month >= 7) & (ttime.month <= 9)], ["stubble", "peak_winter", "pre_monsoon", "monsoon"], "other")
P["season"] = season
trV = met.loc[(times.year <= 2022)[met.index.values], "V"]
q1, q2 = np.nanquantile(np.log(trV), [1 / 3, 2 / 3])
lv = np.log(met.loc[tt, "V"].values)
P["vent"] = np.select([lv < q1, lv < q2], ["low", "mid"], "high")
# transport ratio: s100 * h / distance to nearest upwind station within +-45 deg
lat0, lon0 = lat.mean(), lon.mean()
xk = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0)); yk = (lat - lat0) * 110.57
u = met.loc[tt, "u100"].values; v = met.loc[tt, "v100"].values; s = np.hypot(u, v)
jj = P.j.values
dx = xk[jj][:, None] - xk[None, :]; dy = yk[jj][:, None] - yk[None, :]   # vector source i -> receiver j
d = np.hypot(dx, dy); d[np.arange(len(jj)), jj] = np.inf
cosang = (u[:, None] * dx + v[:, None] * dy) / (np.maximum(s, 1e-3)[:, None] * np.maximum(d, 1e-6))
dup = np.where((cosang >= np.cos(np.deg2rad(45))) & np.isfinite(d), d, np.inf).min(1)
rho = s * 3.6 * P.h.values / dup
P["rho_bin"] = np.select([~np.isfinite(dup), rho < 0.5, rho <= 2], ["no_upwind", "<0.5", "0.5-2"], ">2")
P["date"] = ttime.normalize(); P["week"] = ttime.to_period("W").astype(str)

# dm_hln: Diebold-Mariano test with Harvey-Leybourne-Newbold correction on daily loss differences
# nw_tstat: t-test with Newey-West standard errors
# block_boot_ci: 95% CI of an MAE difference, resampling whole ISO weeks
# ---------- helpers ----------
def dm_hln(d, h_lag=1):
    d = d[np.isfinite(d)]; n = len(d); dbar = d.mean()
    g = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h_lag)]
    var = (g[0] + 2 * sum(g[1:])) / n
    stat = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * h_lag + h_lag * (h_lag - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), n - 1)

def nw_tstat(x, lag=7):
    x = x[np.isfinite(x)]; n = len(x); xb = x.mean(); e = x - xb
    s2 = e @ e / n + 2 * sum((1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, lag + 1))
    t = xb / np.sqrt(s2 / n); return xb, t, 2 * stats.t.sf(abs(t), n - 1)

def block_boot_ci(df, a, b, B=1000):
    """95% CI of MAE(a) - MAE(b), resampling ISO weeks."""
    g = df.assign(ea=(df[a] - df.y).abs(), eb=(df[b] - df.y).abs()).groupby("week")[["ea", "eb"]].agg(["sum", "count"])
    sa, sb, n = g[("ea", "sum")].values, g[("eb", "sum")].values, g[("ea", "count")].values
    W = len(n); res = []
    for _ in range(B):
        k = rng.integers(0, W, W); res.append(sa[k].sum() / n[k].sum() - sb[k].sum() / n[k].sum())
    return np.percentile(res, [2.5, 97.5])

# Overall accuracy by horizon (MAE, RMSE, R2), each arm vs. the no-graph arm, Holm-adjusted.
# ---------- overall table + DM tests ----------
rows = []
for h, g in P.groupby("h"):
    m0 = (g[base_arm] - g.y).abs().mean()
    for a in arms + others:
        mae = (g[a] - g.y).abs().mean(); rmse = float(np.sqrt(((g[a] - g.y) ** 2).mean()))
        r2 = float(1 - ((g[a] - g.y) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum())
        row = dict(h=h, model=a, mae=mae, rmse=rmse, r2=r2, pct_vs_G0=100 * (mae - m0) / m0)
        if a != base_arm:
            daily = g.assign(e=(g[a] - g.y).abs() - (g[base_arm] - g.y).abs()).groupby("date").e.mean().values
            st, p = dm_hln(daily); lo, hi = block_boot_ci(g, a, base_arm)
            row.update(dm_stat=st, dm_p=p, ci_lo=lo, ci_hi=hi)
        rows.append(row)
R = pd.DataFrame(rows)
prim = R[R.model.isin([a for a in arms if a != base_arm])].copy()
order = prim.dm_p.sort_values().index; m = len(order); holm = pd.Series(index=order, dtype=float); run = 0
for r, ix in enumerate(order): run = max(run, min(1, (m - r) * prim.loc[ix, "dm_p"])); holm[ix] = run
R.loc[holm.index, "dm_p_holm"] = holm
R.to_csv(os.path.join(out, f"{tag}_overall.csv"), index=False)

# wind vs uniform/distance (does direction add anything beyond "use the neighbours"?)
pair_rows = []
for h, g in P.groupby("h"):
    for a, b in [("G5_hourly_wind", "G2_distance"), ("G5_hourly_wind", "G1_uniform"), ("G6_vent_gated", "G2_distance"),
                 ("G5_hourly_wind", "G4_static_wind"), ("G4_static_wind", "G2_distance")]:
        if a in g and b in g:
            daily = g.assign(e=(g[a] - g.y).abs() - (g[b] - g.y).abs()).groupby("date").e.mean().values
            st, p = dm_hln(daily); lo, hi = block_boot_ci(g, a, b)
            mb = (g[b] - g.y).abs().mean(); ma = (g[a] - g.y).abs().mean()
            pair_rows.append(dict(h=h, a=a, b=b, mae_a=ma, mae_b=mb, pct=100 * (ma - mb) / mb, dm_p=p, ci_lo=lo, ci_hi=hi))
pd.DataFrame(pair_rows).to_csv(os.path.join(out, f"{tag}_pairwise.csv"), index=False)

# ---------- regime tables ----------
reg_rows = []
for col in ["season", "vent", "rho_bin"]:
    for (h, lev), g in P.groupby(["h", col]):
        m0 = (g[base_arm] - g.y).abs().mean()
        r = dict(h=h, regime=col, level=lev, n=len(g), mae_G0=m0)
        for a in arms:
            if a == base_arm: continue
            r[f"pct_{a}"] = 100 * ((g[a] - g.y).abs().mean() - m0) / m0
        if "G5_hourly_wind" in g and "G2_distance" in g:
            r["pct_G5_vs_G2"] = 100 * ((g.G5_hourly_wind - g.y).abs().mean() / (g.G2_distance - g.y).abs().mean() - 1)
        if "persistence" in g: r["pct_persist"] = 100 * ((g.persistence - g.y).abs().mean() - m0) / m0
        reg_rows.append(r)
RG = pd.DataFrame(reg_rows); RG.to_csv(os.path.join(out, f"{tag}_regimes.csv"), index=False)

# ---------- H3: is the wind-graph gain larger at intermediate transport ratio? ----------
h3 = []
for comp in [("G5_hourly_wind", "G0_none"), ("G5_hourly_wind", "G2_distance")]:
    a, b = comp
    if a not in P or b not in P: continue
    for h, g in P.groupby("h"):
        g = g.assign(e=(g[a] - g.y).abs() - (g[b] - g.y).abs())
        mid = g[g.rho_bin == "0.5-2"].groupby("date").e.mean(); rest = g[g.rho_bin != "0.5-2"].groupby("date").e.mean()
        dd = (mid - rest).dropna().values
        if len(dd) < 20: h3.append(dict(a=a, b=b, h=h, n_days=len(dd))); continue
        xb, t, p = nw_tstat(dd)
        h3.append(dict(a=a, b=b, h=h, n_days=len(dd), mean_mid_minus_rest=xb, t=t, p=p,
                       share_mid=float((g.rho_bin == "0.5-2").mean())))
pd.DataFrame(h3).to_csv(os.path.join(out, f"{tag}_H3_conditional.csv"), index=False)

# ---------- figures ----------
fig, ax = plt.subplots(figsize=(6.2, 3.2))
HS = sorted(P.h.unique()); w = 0.8 / (len(arms) - 1)
for k, a in enumerate([x for x in arms if x != base_arm]):
    vals = [R[(R.h == h) & (R.model == a)].pct_vs_G0.values[0] for h in HS]
    ax.bar(np.arange(len(HS)) + k * w - 0.4 + w / 2, vals, w, label=a.split("_", 1)[1].replace("_", " "))
ax.axhline(0, c="k", lw=0.8); ax.set_xticks(range(len(HS))); ax.set_xticklabels([f"{h} h" for h in HS])
ax.set_ylabel("MAE change vs. no graph (%)"); ax.legend(fontsize=6.5, ncol=2, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(out, f"{tag}_fig_overall.png"), dpi=220)

if "G5_hourly_wind" in P and "G2_distance" in P:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    sub = RG[RG.regime == "rho_bin"]
    for lev, c in zip(["<0.5", "0.5-2", ">2", "no_upwind"], ["#9ecae1", "#1f5fa8", "#08306b", "#bbbbbb"]):
        q = sub[sub.level == lev].sort_values("h")
        ax.plot(q.h, q.pct_G5_vs_G2, marker="o", c=c, label=f"rho {lev}")
    ax.axhline(0, c="k", lw=0.8); ax.set_xlabel("horizon (h)"); ax.set_ylabel("MAE: hourly wind vs distance (%)")
    ax.legend(fontsize=7, frameon=False); plt.tight_layout(); plt.savefig(os.path.join(out, f"{tag}_fig_rho.png"), dpi=220)

print(R.round(3).to_string())
print(pd.DataFrame(pair_rows).round(3).to_string())
print(pd.DataFrame(h3).round(4).to_string())
