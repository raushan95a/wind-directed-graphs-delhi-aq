"""Step 3: controlled comparison of graph constructions with a fixed learner (XGBoost, GPU on Colab).

The graph enters ONLY through neighbour-aggregate features m_j(t-l) = sum_i P_ji x_i(t-l) / sum_i P_ji
(over observed neighbours) plus the edge mass sum_i P_ji. Everything else is identical across arms.
Meteorology is used only at the forecast origin t (no future weather).

Usage: python graph_experiment.py <prep_dir> <out_dir> [test_year]
"""
import sys, os, json, time
import numpy as np, pandas as pd, xgboost as xgb
try:
    import torch; DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
print("xgboost", xgb.__version__, "device", DEVICE, flush=True)

prep, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
# =====================================================================================
# 0) Load the hourly panel and define the rolling-origin split for the chosen test year.
# =====================================================================================
Z = np.load(os.path.join(prep, "panel.npz"), allow_pickle=True)
X = Z["pm"].astype(np.float32); lat, lon = Z["lat"], Z["lon"]
times = pd.to_datetime(Z["times"].astype("datetime64[h]")); T, N = X.shape
HORIZONS = [1, 3, 6, 12, 24]; NLAGS = [0, 1, 2, 3, 6]
yr = times.year.values
TEST_YEAR = int(sys.argv[3]) if len(sys.argv) > 3 else 2024   # rolling origin: train < TEST-1, val = TEST-1
train_o = (yr <= TEST_YEAR - 2) & (np.arange(T) >= 24); val_o = yr == TEST_YEAR - 1; test_o = yr == TEST_YEAR

# ---------------- meteorology at origin (city mean over ERA5 grid points) ----------------
met = pd.read_parquet(os.path.join(prep, "met.parquet"))
def uv(s, d): r = np.deg2rad(d); return -s * np.sin(r), -s * np.cos(r)
met["u100"], met["v100"] = uv(met.wind_speed_100m, met.wind_direction_100m)
met["u10"], met["v10"] = uv(met.wind_speed_10m, met.wind_direction_10m)
M = met.groupby("time")[["u100", "v100", "u10", "v10", "boundary_layer_height"]].mean().reindex(times).interpolate(limit=3)
u, v = M.u100.values, M.v100.values; s100 = np.hypot(u, v); s10 = np.hypot(M.u10.values, M.v10.values)
blh = M.boundary_layer_height.values; V = np.maximum(blh, 50) * np.maximum(s10, 0.1)
lnV0 = np.nanmedian(np.log(V[train_o]))

# ---------------- geometry ----------------
lat0, lon0 = lat.mean(), lon.mean()
xk = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0)); yk = (lat - lat0) * 110.57
DX = xk[None, :] - xk[:, None]; DY = yk[None, :] - yk[:, None]          # [i, j]: vector i -> j
D = np.hypot(DX, DY); np.fill_diagonal(D, np.inf)
EX, EY = DX / D, DY / D                                                  # unit bearing i -> j
def K(sig): k = np.exp(-(D ** 2) / (2 * sig ** 2)); return k            # diag -> 0 (D=inf)

# correlation graph (train only): top-5 correlated neighbours
C = pd.DataFrame(X[train_o]).corr().values; np.fill_diagonal(C, -np.inf)
Pcorr = np.zeros((N, N), np.float32)
for j in range(N):
    for i in np.argsort(C[:, j])[-5:]: Pcorr[j, i] = max(C[i, j], 0)

def calm(s): return 1 / (1 + np.exp(-(s - 1.0) / 0.5))

# -------------------------------------------------------------------------------------
# Neighbour aggregation for a fixed graph P (receiver j x source i): weighted mean of the
# neighbours' PM2.5 at lags 0,1,2,3,6 h, skipping missing values, plus the edge mass.
# -------------------------------------------------------------------------------------
def static_agg(P):
    """P[j, i] receiver j, source i. returns agg[T, N, len(NLAGS)], mass[N]"""
    mask = np.isfinite(X).astype(np.float32); X0 = np.nan_to_num(X)
    feats = []
    for l in NLAGS:
        Xl = np.roll(X0, l, 0); Ml = np.roll(mask, l, 0); Xl[:l] = 0; Ml[:l] = 0
        num = Xl @ P.T; den = Ml @ P.T
        feats.append(np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan))
    return np.stack(feats, -1), np.repeat(P.sum(1)[None, :], T, 0)

# Hourly wind graph: distance kernel x positive cosine between wind and bearing x calm gate.
# With 'gate', mix it with the distance graph according to ventilation (stagnant air -> distance).
def dynamic_wind_P(sig, gate=None):
    """P_t[j, i] = K_ij * relu(cos(wind_t, bearing i->j)) * calm(s_t); optional hybrid with distance."""
    cosang = (u[:, None, None] * EX[None] + v[:, None, None] * EY[None]) / np.maximum(s100, 1e-3)[:, None, None]
    Pw = K(sig)[None] * np.maximum(cosang, 0) * calm(s100)[:, None, None]   # [t, i, j]
    Pw = np.transpose(Pw, (0, 2, 1)).astype(np.float32)                        # [t, j, i]
    if gate is not None:
        g = gate[:, None, None]
        Pw = g * Pw + (1 - g) * K(sig).T[None].astype(np.float32)
    return np.nan_to_num(Pw)

# Same aggregation as static_agg, but with a different graph at every forecast origin t.
def dynamic_agg(Pt):
    mask = np.isfinite(X).astype(np.float32); X0 = np.nan_to_num(X)
    feats = []
    for l in NLAGS:
        Xl = np.roll(X0, l, 0); Ml = np.roll(mask, l, 0); Xl[:l] = 0; Ml[:l] = 0
        num = np.einsum("tji,ti->tj", Pt, Xl); den = np.einsum("tji,ti->tj", Pt, Ml)
        feats.append(np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan))
    return np.stack(feats, -1), Pt.sum(2)

# =====================================================================================
# 1) Features shared by every arm. Only the neighbour-aggregate block differs between arms.
# =====================================================================================
# ---------------- base (graph-free) features ----------------
def lagged(A, l): B = np.roll(A, l, 0); B[:l] = np.nan; return B
own = np.stack([lagged(X, l) for l in [0, 1, 2, 3, 4, 5, 11, 23]], -1)
roll24 = pd.DataFrame(X).rolling(24, min_periods=12).mean().values
cal_hour = times.hour.values; cal_month = times.month.values; cal_dow = times.dayofweek.values
metf = np.stack([u, v, s100, s10, np.log(np.maximum(blh, 1)), np.log(V)], -1)

# Build the design matrix for a set of forecast origins and horizon h.
# Columns: own lags | 24 h mean | calendar + station id | origin meteorology | [graph block]
def build(idx, h, gfeat=None, gmass=None):
    tt = np.repeat(idx, N); jj = np.tile(np.arange(N), len(idx))
    tgt = tt + h; okm = tgt < T
    tt, jj, tgt = tt[okm], jj[okm], tgt[okm]
    y = X[tgt, jj]
    cols = [own[tt, jj], roll24[tt, jj][:, None],
            np.stack([(cal_hour[tt] + h) % 24, cal_month[tt], cal_dow[tt], jj], -1).astype(np.float32),
            metf[tt]]
    if gfeat is not None:
        cols += [gfeat[tt, jj], gmass[tt, jj][:, None]]
    F = np.concatenate(cols, 1).astype(np.float32)
    keep = np.isfinite(y) & np.isfinite(F[:, 0])
    return F[keep], y[keep], tt[keep], jj[keep]

# 2) Fixed learner: XGBoost, absolute-error loss, GPU histogram method.
PARAMS = dict(objective="reg:absoluteerror", eta=0.08, max_depth=8, min_child_weight=50, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", device=DEVICE, max_bin=256, seed=0)
tr_idx = np.where(train_o)[0][::2]; va_idx = np.where(val_o)[0]; te_idx = np.where(test_o)[0]

# Train on the change from the current value (so every arm starts at persistence), stop early on
# the validation year, and return test predictions in original units plus the validation MAE.
def fit_eval(gfeat, gmass, h, idx_eval):
    Ftr, ytr, _, _ = build(tr_idx, h, gfeat, gmass); Fva, yva, _, _ = build(va_idx, h, gfeat, gmass)
    # target = change from the station's current value, so every arm starts from persistence
    dtr = xgb.DMatrix(Ftr, ytr - Ftr[:, 0]); dva = xgb.DMatrix(Fva, yva - Fva[:, 0])
    m = xgb.train(PARAMS, dtr, 4000, evals=[(dva, "val")], early_stopping_rounds=150, verbose_eval=False)
    Fe, ye, te, je = build(idx_eval, h, gfeat, gmass)
    rng_ = (0, m.best_iteration + 1)
    pe = Fe[:, 0] + m.predict(xgb.DMatrix(Fe), iteration_range=rng_)
    pv = Fva[:, 0] + m.predict(dva, iteration_range=rng_)
    return m, pe, ye, te, je, float(np.mean(np.abs(pv - yva)))

log = {}
t0 = time.time()
# =====================================================================================
# 3) Graph hyperparameters chosen on the validation year only (6 h horizon).
# ---- tune sigma for distance graph and gate slope on validation (h=6) ----
sig_scores = {}
for sig in [5, 10, 20]:
    gf, gm = static_agg(K(sig).T.astype(np.float32))
    sig_scores[sig] = fit_eval(gf, gm, 6, va_idx)[5]
SIG = min(sig_scores, key=sig_scores.get); log["sigma_val_mae"] = sig_scores; log["sigma"] = SIG
print("sigma val MAE", sig_scores, "->", SIG, f"{time.time()-t0:.0f}s", flush=True)

gate_scores = {}
for a in [1.0, 3.0]:
    g = 1 / (1 + np.exp(-a * (np.log(V) - lnV0)))
    gf, gm = dynamic_agg(dynamic_wind_P(SIG, gate=np.nan_to_num(g, nan=0.5)))
    gate_scores[a] = fit_eval(gf, gm, 6, va_idx)[5]
A_GATE = min(gate_scores, key=gate_scores.get); log["gate_val_mae"] = gate_scores; log["gate_a"] = A_GATE
print("gate val MAE", gate_scores, "->", A_GATE, f"{time.time()-t0:.0f}s", flush=True)

# =====================================================================================
# 4) The seven graph constructions compared in the paper (G0 = no graph).
# =====================================================================================
# ---- build arms ----
ubar, vbar = np.nanmean(u[train_o]), np.nanmean(v[train_o]); sbar = np.hypot(ubar, vbar)
Pstatic = (K(SIG) * np.maximum((ubar * EX + vbar * EY) / sbar, 0)).T.astype(np.float32)
gate = np.nan_to_num(1 / (1 + np.exp(-A_GATE * (np.log(V) - lnV0))), nan=0.5)
arms = {"G0_none": (None, None),
        "G1_uniform": static_agg(np.ones((N, N), np.float32) - np.eye(N, dtype=np.float32)),
        "G2_distance": static_agg(K(SIG).T.astype(np.float32)),
        "G3_correlation": static_agg(Pcorr),
        "G4_static_wind": static_agg(Pstatic),
        "G5_hourly_wind": dynamic_agg(dynamic_wind_P(SIG)),
        "G6_vent_gated": dynamic_agg(dynamic_wind_P(SIG, gate=gate))}
log["static_wind_dir_towards_deg"] = float(np.degrees(np.arctan2(ubar, vbar)) % 360)

# 5) Train and evaluate every arm at every horizon; baselines are added to the same table.
rows = []; preds = []
for h in HORIZONS:
    base = None
    for name, (gf, gm) in arms.items():
        m, p, y, tt, jj, vmae = fit_eval(gf, gm, h, te_idx)
        if base is None:
            base = pd.DataFrame({"t": tt, "j": jj, "h": h, "y": y})
            base["persistence"] = X[tt, jj]
            base["seasonal_naive"] = X[tt + h - 24, jj] if h <= 24 else np.nan
        base[name] = p
        rows.append(dict(h=h, arm=name, test_mae=float(np.mean(np.abs(p - y))), val_mae=vmae, n_trees=int(m.best_iteration)))
        print(h, name, round(rows[-1]["test_mae"], 3), f"{time.time()-t0:.0f}s", flush=True)
    preds.append(base)
P = pd.concat(preds, ignore_index=True)
P.to_parquet(os.path.join(out, "test_predictions.parquet"), index=False)
pd.DataFrame(rows).to_csv(os.path.join(out, "summary_mae.csv"), index=False)
# met at origin for regime analysis
pd.DataFrame({"t": np.arange(T), "time": times, "u100": u, "v100": v, "s100": s100, "s10": s10, "blh": blh, "V": V,
              "gate": gate}).to_parquet(os.path.join(out, "met_origin.parquet"), index=False)
np.savez(os.path.join(out, "geometry.npz"), EX=EX, EY=EY, D=np.where(np.isinf(D), 0, D), lnV0=lnV0)
json.dump(log, open(os.path.join(out, "tuning.json"), "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
