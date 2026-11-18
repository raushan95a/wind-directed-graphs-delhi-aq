"""Step 5 (run on Google Colab, GPU): robustness check with a compact Graph WaveNet-style model.

Same data, splits, horizons and graph constructions as the XGBoost experiment; the only thing
that changes between arms is the support matrix P (receiver x source) used for message passing.

Usage (Colab): python gnn_experiment.py <panel.npz> <met_origin.parquet> <tuning.json> <out_dir>
"""
import os, sys, json, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev)
PANEL, METF, TUNF, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
Z = np.load(PANEL, allow_pickle=True)
X = Z["pm"].astype(np.float32); lat, lon = Z["lat"], Z["lon"]
times = pd.to_datetime(Z["times"].astype("datetime64[h]")); T, N = X.shape
M = pd.read_parquet(METF)
u, v, s100, s10, blh, Vv, gate = [M[c].values.astype(np.float32) for c in ["u100", "v100", "s100", "s10", "blh", "V", "gate"]]
HOR = [1, 3, 6, 12, 24]; L = 24; yr = times.year.values
SIG = float(json.load(open(TUNF))["sigma"])

# =====================================================================================
# 1) Graphs - identical definitions to graph_experiment.py; only the support P changes per arm.
# =====================================================================================
lat0, lon0 = lat.mean(), lon.mean()
xk = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0)); yk = (lat - lat0) * 110.57
DX = xk[None, :] - xk[:, None]; DY = yk[None, :] - yk[:, None]; D = np.hypot(DX, DY); np.fill_diagonal(D, np.inf)
EX, EY = DX / D, DY / D
Kd = np.exp(-(D ** 2) / (2 * SIG ** 2))                     # [i, j], zero diagonal
dK = Kd.T.sum(1, keepdims=True)                              # fixed denominator per receiver
P_dist = (Kd.T / dK).astype(np.float32)                      # [j, i]
TEST_YEAR = int(os.environ.get("TEST_YEAR", "2024"))   # train <= TEST-2, validate TEST-1, test TEST
train_o = (yr <= TEST_YEAR - 2)
ub, vb = np.nanmean(u[train_o]), np.nanmean(v[train_o]); sb = np.hypot(ub, vb)
P_static = ((Kd * np.maximum((ub * EX + vb * EY) / sb, 0)).T / dK).astype(np.float32)
calm = lambda s: 1 / (1 + np.exp(-(s - 1.0) / 0.5))

tEX, tEY = torch.tensor(np.nan_to_num(EX), device=dev, dtype=torch.float32), torch.tensor(np.nan_to_num(EY), device=dev, dtype=torch.float32)
tKd = torch.tensor(np.nan_to_num(Kd), device=dev, dtype=torch.float32); tdK = torch.tensor(dK, device=dev, dtype=torch.float32)
tu, tv = torch.tensor(np.nan_to_num(u), device=dev), torch.tensor(np.nan_to_num(v), device=dev)
ts = torch.tensor(np.nan_to_num(s100), device=dev); tg = torch.tensor(np.nan_to_num(gate, nan=0.5), device=dev)
tPdist = torch.tensor(P_dist, device=dev)

def P_wind(ii, gated=False):
    """ii: LongTensor of origin indices. returns [B, N(receiver), N(source)] on GPU."""
    uu, vv, ss = tu[ii], tv[ii], ts[ii].clamp_min(1e-3)
    cos = (uu[:, None, None] * tEX[None] + vv[:, None, None] * tEY[None]) / ss[:, None, None]
    calm_ = torch.sigmoid((ts[ii] - 1.0) / 0.5)
    Pw = (tKd[None] * cos.clamp_min(0) * calm_[:, None, None]).transpose(1, 2) / tdK[None]
    if gated:
        g = tg[ii][:, None, None]; Pw = g * Pw + (1 - g) * tPdist[None]
    return torch.nan_to_num(Pw)

# 2) Normalise with training statistics only; forward-fill inputs and keep a missing-value mask.
# ---- data tensors ----
mu = np.nanmean(X[train_o]); sd = np.nanstd(X[train_o])
Xn = (X - mu) / sd; mask = np.isfinite(Xn); Xf = pd.DataFrame(Xn).ffill().fillna(0).values.astype(np.float32)
metf = np.stack([u, v, s100, s10, np.log(np.maximum(blh, 1)), np.log(np.maximum(Vv, 1))], -1)
metf = np.nan_to_num((metf - np.nanmean(metf[train_o], 0)) / np.nanstd(metf[train_o], 0)).astype(np.float32)
Xt = torch.tensor(Xf, device=dev); Mt = torch.tensor(mask.astype(np.float32), device=dev)
Yraw = torch.tensor(np.nan_to_num(X, nan=-1.0), device=dev); METt = torch.tensor(metf, device=dev)
hourt = torch.tensor(times.hour.values, device=dev, dtype=torch.long)

def origins(sel): return np.where(sel & (np.arange(T) >= L) & (np.arange(T) < T - max(HOR)))[0]
tr, va, te = origins(yr <= TEST_YEAR - 2), origins(yr == TEST_YEAR - 1), origins(yr == TEST_YEAR)

# =====================================================================================
# 3) Compact Graph WaveNet-style model:
#    dilated gated temporal convolutions -> origin meteorology + hour -> two forward-only
#    diffusion layers Z' = Z W0 + P Z W1 + P^2 Z W2 -> per-station head for all horizons.
# =====================================================================================
class Net(nn.Module):
    def __init__(s, n, hid=48, adaptive=False):
        super().__init__()
        s.inp = nn.Conv1d(2, hid, 1)
        s.tcn = nn.ModuleList([nn.Conv1d(hid, 2 * hid, 2, dilation=d) for d in [1, 2, 4, 8]])
        s.emb = nn.Embedding(n, 8); s.metp = nn.Linear(6 + 24, hid)
        s.g0 = nn.ModuleList([nn.Linear(hid, hid) for _ in range(2)])
        s.g1 = nn.ModuleList([nn.Linear(hid, hid) for _ in range(2)])
        s.g2 = nn.ModuleList([nn.Linear(hid, hid) for _ in range(2)])
        s.out = nn.Sequential(nn.Linear(hid + 8, 64), nn.ReLU(), nn.Linear(64, len(HOR)))
        s.adaptive = adaptive
        if adaptive: s.E1 = nn.Parameter(torch.randn(n, 10) * 0.1); s.E2 = nn.Parameter(torch.randn(n, 10) * 0.1)
    def forward(s, x, m, met, hr, P):
        B, Lx, n = x.shape
        h = s.inp(torch.stack([x, m], 2).permute(0, 3, 2, 1).reshape(B * n, 2, Lx))
        for c in s.tcn:
            z = c(F.pad(h, (c.dilation[0], 0))); a, b = z.chunk(2, 1); h = h + torch.tanh(a) * torch.sigmoid(b)
        h = h[:, :, -1].reshape(B, n, -1)
        h = h + s.metp(torch.cat([met, F.one_hot(hr, 24).float()], -1))[:, None, :]
        if s.adaptive: P = torch.softmax(F.relu(s.E1 @ s.E2.T), 1)[None].expand(B, n, n)
        for k in range(2):
            p1 = torch.bmm(P, h); p2 = torch.bmm(P, p1)
            h = F.relu(s.g0[k](h) + s.g1[k](p1) + s.g2[k](p2)) + h
        e = s.emb.weight[None].expand(B, n, 8)
        return s.out(torch.cat([h, e], -1))  # [B, n, H] in normalised units

# Assemble one mini-batch: 24 h input windows, mask, origin meteorology, graph P and targets.
def batch(idx, arm):
    ii = torch.tensor(idx, device=dev)
    win = ii[:, None] + torch.arange(-L + 1, 1, device=dev)[None]
    x, m = Xt[win], Mt[win]; met = METt[ii]; hr = hourt[ii]
    if arm == "G0_none" or arm == "G7_adaptive": P = torch.zeros(len(idx), N, N, device=dev)
    elif arm == "G2_distance": P = torch.tensor(P_dist, device=dev)[None].expand(len(idx), N, N)
    elif arm == "G4_static_wind": P = torch.tensor(P_static, device=dev)[None].expand(len(idx), N, N)
    elif arm == "G5_hourly_wind": P = P_wind(ii)
    elif arm == "G6_vent_gated": P = P_wind(ii, gated=True)
    y = torch.stack([Yraw[ii + h] for h in HOR], -1)  # [B, n, H]
    return x, m, met, hr, P, y

# Predict in original units for a set of origins (no gradient).
def evaluate(model, idx, arm):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(idx), 512):
            x, m, met, hr, P, y = batch(idx[b:b + 512], arm)
            out.append((model(x, m, met, hr, P) * sd + mu).cpu().numpy())
    return np.concatenate(out)

# =====================================================================================
# 4) Train every graph arm with several seeds; early stopping on the validation year.
# =====================================================================================
ARMS = ["G0_none", "G2_distance", "G4_static_wind", "G5_hourly_wind", "G6_vent_gated", "G7_adaptive"]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]; rows = []; allpred = {}
Yte = np.stack([X[te + h] for h in HOR], -1)  # [n_te, N, H]
for arm in ARMS:
    preds = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); t0 = time.time()
        model = Net(N, adaptive=(arm == "G7_adaptive")).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
        best, bad, state = 1e9, 0, None
        for ep in range(40):
            model.train(); perm = np.random.permutation(tr)
            for b in range(0, len(perm), 256):
                x, m, met, hr, P, y = batch(perm[b:b + 256], arm)
                pr = model(x, m, met, hr, P) * sd + mu; ok = y >= 0
                loss = (pr - y).abs()[ok].mean() / 50
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
            pv = evaluate(model, va, arm); Yv = np.stack([X[va + h] for h in HOR], -1)
            vm = np.nanmean(np.abs(pv - Yv))
            if vm < best - 1e-3: best, bad, state = vm, 0, {k: x.clone() for k, x in model.state_dict().items()}
            else: bad += 1
            if bad >= 5: break
        model.load_state_dict(state); p = evaluate(model, te, arm); preds.append(p)
        for hi, h in enumerate(HOR):
            rows.append(dict(arm=arm, seed=seed, h=h, val_mae=best, test_mae=float(np.nanmean(np.abs(p[..., hi] - Yte[..., hi])))))
        print(arm, seed, "val", round(best, 2), "epochs", ep + 1, f"{time.time()-t0:.0f}s", flush=True)
    allpred[arm] = np.mean(preds, 0)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "gnn_summary.csv"), index=False)
recs = []
for hi, h in enumerate(HOR):
    tt = np.repeat(te, N); jj = np.tile(np.arange(N), len(te)); y = Yte[..., hi].ravel()
    d = pd.DataFrame({"t": tt, "j": jj, "h": h, "y": y})
    for arm in ARMS: d[arm] = allpred[arm][..., hi].ravel()
    recs.append(d[np.isfinite(y)])
pd.concat(recs).to_parquet(os.path.join(OUT, "gnn_test_predictions.parquet"), index=False)
print(pd.DataFrame(rows).groupby(["arm", "h"]).test_mae.agg(["mean", "std"]).round(2))
