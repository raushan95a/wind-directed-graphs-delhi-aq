"""Step 6: training dynamics and resource consumption (for the paper's required graphs).

Reuses the exact definitions of graph_experiment.py and gnn_experiment.py, then for one test year (default 2024):
  * XGBoost (h = 6 h): validation-MAE learning curves, graph-construction time, fit time, inference time.
  * GNN (1 seed): per-epoch training loss and validation MAE, time per epoch, peak GPU memory,
    parameter count and inference latency, for every graph arm.
Usage: python resources.py <prep_dir> <xgb_out_dir_for_test_year> <out_dir> [test_year=2024]
"""
import sys, os, json, time
prep, xgbdir, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
TY = sys.argv[4] if len(sys.argv) > 4 else "2024"; os.environ["TEST_YEAR"] = TY
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Re-use the exact definitions from graph_experiment.py (executed up to the training loop).
# ======================= XGBoost =======================
src = open(os.path.join(HERE, "graph_experiment.py"), encoding="utf-8").read().split("log = {}")[0]
sys.argv = ["03", prep, os.path.join(out, "_tmp"), TY]
g = {"__name__": "xgb_defs", "__file__": os.path.join(HERE, "graph_experiment.py")}
exec(src, g)
xgb = g["xgb"]; tune = json.load(open(os.path.join(xgbdir, "tuning.json")))
SIG, A = tune["sigma"], tune["gate_a"]
V, lnV0, u, v, train_o = g["V"], g["lnV0"], g["u"], g["v"], g["train_o"]
gate = np.nan_to_num(1 / (1 + np.exp(-A * (np.log(V) - lnV0))), nan=0.5)
ubar, vbar = np.nanmean(u[train_o]), np.nanmean(v[train_o]); sbar = np.hypot(ubar, vbar)
K, EX, EY, N = g["K"], g["EX"], g["EY"], g["N"]
builders = {
    "G0_none": lambda: (None, None),
    "G1_uniform": lambda: g["static_agg"](np.ones((N, N), np.float32) - np.eye(N, dtype=np.float32)),
    "G2_distance": lambda: g["static_agg"](K(SIG).T.astype(np.float32)),
    "G3_correlation": lambda: g["static_agg"](g["Pcorr"]),
    "G4_static_wind": lambda: g["static_agg"]((K(SIG) * np.maximum((ubar * EX + vbar * EY) / sbar, 0)).T.astype(np.float32)),
    "G5_hourly_wind": lambda: g["dynamic_agg"](g["dynamic_wind_P"](SIG)),
    "G6_vent_gated": lambda: g["dynamic_agg"](g["dynamic_wind_P"](SIG, gate=gate)),
}
H = 6; xres, curves = [], {}
for name, fn in builders.items():
    t0 = time.time(); gf, gm = fn(); t_graph = time.time() - t0
    Ftr, ytr, _, _ = g["build"](g["tr_idx"], H, gf, gm); Fva, yva, _, _ = g["build"](g["va_idx"], H, gf, gm)
    Fte, yte, _, _ = g["build"](g["te_idx"], H, gf, gm)
    dtr = xgb.DMatrix(Ftr, ytr - Ftr[:, 0]); dva = xgb.DMatrix(Fva, yva - Fva[:, 0])
    ev = {}; t0 = time.time()
    m = xgb.train(g["PARAMS"], dtr, 4000, evals=[(dtr, "train"), (dva, "val")], early_stopping_rounds=150,
                  evals_result=ev, verbose_eval=False)
    t_fit = time.time() - t0
    dte = xgb.DMatrix(Fte); t0 = time.time()
    p = Fte[:, 0] + m.predict(dte, iteration_range=(0, m.best_iteration + 1)); t_inf = time.time() - t0
    curves[name] = {"train": ev["train"]["mae"], "val": ev["val"]["mae"]}
    xres.append(dict(arm=name, n_features=Ftr.shape[1], graph_build_s=t_graph, fit_s=t_fit, best_iter=int(m.best_iteration),
                     infer_s=t_inf, infer_us_per_sample=1e6 * t_inf / len(yte), test_mae=float(np.mean(np.abs(p - yte)))))
    print("xgb", xres[-1], flush=True)
pd.DataFrame(xres).to_csv(os.path.join(out, "xgb_resources_h6.csv"), index=False)
json.dump(curves, open(os.path.join(out, "xgb_curves_h6.json"), "w"))
del g

# Re-use the exact definitions from gnn_experiment.py (executed up to the training loop).
# ======================= GNN =======================
import torch, torch.nn as nn
src = open(os.path.join(HERE, "gnn_experiment.py"), encoding="utf-8").read().split("ARMS = [")[0]
sys.argv = ["05", os.path.join(prep, "panel.npz"), os.path.join(xgbdir, "met_origin.parquet"), os.path.join(xgbdir, "tuning.json"), os.path.join(out, "_tmp")]
G = {"__name__": "gnn_defs", "__file__": os.path.join(HERE, "gnn_experiment.py")}
exec(src, G)
dev, Net, batch, evaluate, X, HOR = G["dev"], G["Net"], G["batch"], G["evaluate"], G["X"], G["HOR"]
tr, va, te, mu, sd, Nn = G["tr"], G["va"], G["te"], G["mu"], G["sd"], G["N"]
gres, hist = [], {}
Yv = np.stack([X[va + h] for h in HOR], -1)
for arm in ["G0_none", "G2_distance", "G4_static_wind", "G5_hourly_wind", "G6_vent_gated", "G7_adaptive"]:
    torch.manual_seed(0); np.random.seed(0)
    if dev == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    model = Net(Nn, adaptive=(arm == "G7_adaptive")).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    h_tr, h_va, h_t = [], [], []; best, bad = 1e9, 0
    for ep in range(40):
        model.train(); perm = np.random.permutation(tr); t0 = time.time(); tot, nb = 0.0, 0
        for b in range(0, len(perm), 256):
            x, m_, met, hr, P, y = batch(perm[b:b + 256], arm)
            pr = model(x, m_, met, hr, P) * sd + mu; ok = y >= 0
            loss = (pr - y).abs()[ok].mean() / 50
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
            tot += float(loss) * 50; nb += 1
        if dev == "cuda": torch.cuda.synchronize()
        h_t.append(time.time() - t0); h_tr.append(tot / nb)
        vm = float(np.nanmean(np.abs(evaluate(model, va, arm) - Yv))); h_va.append(vm)
        if vm < best - 1e-3: best, bad = vm, 0
        else: bad += 1
        if bad >= 5: break
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else float("nan")
    xb = batch(te[:512], arm)
    with torch.no_grad():
        for _ in range(3): model(*xb[:5])
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): model(*xb[:5])
        if dev == "cuda": torch.cuda.synchronize()
    lat = (time.time() - t0) / 20
    hist[arm] = {"train_mae": h_tr, "val_mae": h_va, "epoch_s": h_t}
    gres.append(dict(arm=arm, params=nparam, epochs=len(h_tr), mean_epoch_s=float(np.mean(h_t)), peak_gpu_mb=peak,
                     infer_ms_per_batch512=1000 * lat, infer_us_per_station_forecast=1e6 * lat / (512 * Nn), best_val_mae=best))
    print("gnn", gres[-1], flush=True)
pd.DataFrame(gres).to_csv(os.path.join(out, "gnn_resources.csv"), index=False)
json.dump(hist, open(os.path.join(out, "gnn_curves.json"), "w"))

# ======================= figures =======================
lab = lambda a: a.split("_", 1)[1].replace("_", " ")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for a, c in hist.items(): ax[0].plot(range(1, len(c["val_mae"]) + 1), c["val_mae"], label=lab(a))
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation MAE (ug/m3)"); ax[0].set_title("GNN"); ax[0].legend(fontsize=6, frameon=False)
for a, c in curves.items(): ax[1].plot(c["val"], label=lab(a), lw=1)
ax[1].set_xlabel("boosting round"); ax[1].set_ylabel("validation MAE of change"); ax[1].set_title("XGBoost, h = 6 h")
plt.tight_layout(); plt.savefig(os.path.join(out, "fig_training_curves.png"), dpi=220)

X_ = pd.DataFrame(xres); G_ = pd.DataFrame(gres)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax[0].barh([lab(a) for a in X_.arm], X_.graph_build_s + X_.fit_s); ax[0].set_xlabel("XGBoost build + fit (s)")
ax[1].barh([lab(a) for a in G_.arm], G_.mean_epoch_s); ax[1].set_xlabel("GNN time / epoch (s)")
ax[2].barh([lab(a) for a in G_.arm], G_.infer_ms_per_batch512); ax[2].set_xlabel("GNN inference / 512 origins (ms)")
for a_ in ax[1:]: a_.set_yticklabels([])
plt.tight_layout(); plt.savefig(os.path.join(out, "fig_resources.png"), dpi=220)
print("done")
