"""Build the paper's results tables (LaTeX) and figures from the Colab outputs.

Usage: python tools/make_paper_tables.py <unzipped_results_dir> [<output_dir>]
  <unzipped_results_dir>: the results.zip downloaded by the notebook, unzipped
  <output_dir>: where to write tables/, figures/ and results_numbers.json (default: paper_tables/)
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = sys.argv[1]; RES = None
HERE = sys.argv[2] if len(sys.argv) > 2 else "paper_tables"
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True); os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "font.family": "serif", "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "figure.dpi": 200})
COLW = 3.45
NAMES = {"G0_none": "No graph", "G1_uniform": "Uniform", "G2_distance": "Distance", "G3_correlation": "Correlation",
         "G4_static_wind": "Static wind", "G5_hourly_wind": "Hourly wind", "G6_vent_gated": "Vent.-gated",
         "G7_adaptive": "Adaptive", "persistence": "Persistence", "seasonal_naive": "Seasonal naive"}
ORDER = ["persistence", "seasonal_naive", "G0_none", "G1_uniform", "G2_distance", "G3_correlation",
         "G4_static_wind", "G5_hourly_wind", "G6_vent_gated"]
COL = {"G1_uniform": "#7f7f7f", "G2_distance": "#1f77b4", "G3_correlation": "#9467bd", "G4_static_wind": "#ff7f0e",
       "G5_hourly_wind": "#d62728", "G6_vent_gated": "#2ca02c", "G7_adaptive": "#8c564b", "G0_none": "#000000"}
nums = {}
def p(*a): return os.path.join(R, *a)
def save(fig, name): fig.savefig(os.path.join(HERE, "figures", name), bbox_inches="tight"); plt.close(fig)
def sg(v): return "0.0" if abs(v) < 0.05 else f"{v:+.1f}"
def star(pv): return "" if not np.isfinite(pv) else ("$^{*}$" if pv < 0.05 else "")

import glob, re
YEARS = sorted([int(d[-4:]) for d in glob.glob(p("stats_test*")) if re.search(r"stats_test\d{4}$", d)
                and os.path.exists(os.path.join(d, f"xgb{d[-4:]}_overall.csv"))], reverse=True)
if RES is None and os.path.exists(p("resources", "gnn_curves.json")): RES = p("resources")
print("test years found:", YEARS)
nums["years"] = YEARS

# ---------------- Table: main (most recent test year: MAE and RMSE blocks) ----------------
rows = []
if YEARS:
    y = YEARS[0]
    O = pd.read_csv(p(f"stats_test{y}", f"xgb{y}_overall.csv"))
    for yy in YEARS: nums[f"overall_{yy}"] = pd.read_csv(p(f"stats_test{yy}", f"xgb{yy}_overall.csv")).to_dict("records")
    HS = sorted(O.h.unique())
    metrics = [("mae", "MAE")]
    for met, lab in metrics:
        rows.append(r"\multicolumn{6}{l}{\textit{%s, test year %d}} \\" % (lab, y))
        best = {h: O[(O.h == h) & O.model.str.startswith("G")][met].min() for h in HS}
        for m in ORDER:
            r = O[O.model == m]
            if r.empty: continue
            cells = []
            for h in HS:
                q = r[r.h == h].iloc[0]; v = f"{q[met]:.2f}"
                if abs(q[met] - best[h]) < 1e-9: v = r"\textbf{" + v + "}"
                if met == "mae" and m.startswith("G") and m != "G0_none": v += star(q.get("dm_p_holm", np.nan))
                cells.append(v)
            rows.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
if rows:
    rows = rows[:-1]
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Test error ($\mu$g\,m$^{-3}$) of each graph construction with the fixed XGBoost learner on the most recent test year. Bold: best graph arm; $^{*}$: MAE differs from the no-graph arm (Diebold--Mariano, Holm-adjusted $p<0.05$).}" "\n"
           r"\label{tab:main}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{4pt}" "\n"
           r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n" r"Model & 1 h & 3 h & 6 h & 12 h & 24 h \\" "\n" r"\midrule" "\n"
           + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(HERE, "tables", "tab_main.tex"), "w").write(tex)

# ---------------- Table: directional ablation (pairwise) ----------------
pairs = [("G5_hourly_wind", "G2_distance", "Hourly wind vs.\\ distance"), ("G4_static_wind", "G2_distance", "Static wind vs.\\ distance"),
         ("G6_vent_gated", "G2_distance", "Vent.-gated vs.\\ distance"),
         ]
rows = []
for y in YEARS:
    f = p(f"stats_test{y}", f"xgb{y}_pairwise.csv")
    if not os.path.exists(f): continue
    Pw = pd.read_csv(f); nums[f"pairwise_{y}"] = Pw.to_dict("records")
    rows.append(r"\multicolumn{6}{l}{\textit{Test year %d}} \\" % y)
    for a, b, lab in pairs:
        r = Pw[(Pw.a == a) & (Pw.b == b)].sort_values("h")
        if r.empty: continue
        rows.append(lab + " & " + " & ".join(sg(q.pct) + star(q.dm_p) for q in r.itertuples()) + r" \\")
    rows.append(r"\midrule")
if rows:
    rows = rows[:-1]
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Ablation of edge direction and timing: relative MAE change (\%) of the first graph against the second. Positive values mean the first graph is worse; $^{*}$: Diebold--Mariano $p<0.05$.}" "\n"
           r"\label{tab:ablation}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3.5pt}" "\n"
           r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n" r"Comparison & 1 h & 3 h & 6 h & 12 h & 24 h \\" "\n" r"\midrule" "\n"
           + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(HERE, "tables", "tab_ablation.tex"), "w").write(tex)

# ---------------- Table: regime / transport-ratio breakdown (2024 and 2023, h = 3) ----------------
rows = []
LV = {"rho_bin": [("<0.5", r"$\rho<0.5$"), ("0.5-2", r"$0.5\le\rho\le2$"), (">2", r"$\rho>2$"), ("no_upwind", "no upwind station")],
      "vent": [("low", "low ventilation"), ("mid", "mid ventilation"), ("high", "high ventilation")],
      "season": [("stubble", "stubble burning"), ("peak_winter", "peak winter"), ("pre_monsoon", "pre-monsoon"), ("monsoon", "monsoon")]}
regs = {}
for y in YEARS[:2]:
    f = p(f"stats_test{y}", f"xgb{y}_regimes.csv")
    if os.path.exists(f): regs[y] = pd.read_csv(f)
if regs:
    nums["regimes"] = {y: d.to_dict("records") for y, d in regs.items()}
    H0 = 3
    for reg, levs in LV.items():
        for lev, lab in levs:
            cells = []
            for y in YEARS[:2]:
                d = regs.get(y)
                q = None if d is None else d[(d.h == H0) & (d.regime == reg) & (d.level == lev)]
                if q is None or q.empty: cells += ["--", "--"]; continue
                q = q.iloc[0]; cells += [sg(q.pct_G5_hourly_wind), sg(q.pct_G5_vs_G2)]
            rows.append(lab + " & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Regime breakdown at the 3~h horizon: MAE change (\%) of the hourly wind graph against the no-graph arm (W/N) and against the distance graph (W/D).}" "\n"
           r"\label{tab:regime}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{4pt}" "\n"
           r"\begin{tabular}{@{}l" + "cc" * len(YEARS[:2]) + r"@{}}" "\n" r"\toprule" "\n" r" & " + " & ".join(r"\multicolumn{2}{c}{%d}" % y for y in YEARS[:2]) + r" \\" "\n"
           r"Regime" + " & W/N & W/D" * len(YEARS[:2]) + r" \\" "\n" r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(HERE, "tables", "tab_regime.tex"), "w").write(tex)

# ---------------- Table: GNN robustness + resource use ----------------
gfiles = sorted(glob.glob(p("gnn_test*", "gnn_summary.csv")))
if gfiles:
    f = gfiles[-1]; gyear = re.findall(r"gnn_test(\d{4})", f)[0]
    Gs = pd.read_csv(f); agg = Gs.groupby(["arm", "h"]).test_mae.agg(["mean", "std"]).reset_index()
    nums["gnn"] = agg.to_dict("records"); nums["gnn_year"] = gyear; nums["gnn_max_std"] = float(agg["std"].max())
    HS = sorted(agg.h.unique()); best = agg.groupby("h")["mean"].min()
    # resource columns
    res_g, res_x = {}, {}
    if RES and os.path.exists(os.path.join(RES, "gnn_resources.csv")):
        Gr = pd.read_csv(os.path.join(RES, "gnn_resources.csv")); res_g = {r.arm: r.mean_epoch_s for r in Gr.itertuples()}
        Xr = pd.read_csv(os.path.join(RES, "xgb_resources_h6.csv")); res_x = {r.arm: r.graph_build_s + r.fit_s for r in Xr.itertuples()}
        rsrc = "measured in a dedicated run"
    elif os.path.exists(p("resources_from_log", "gnn_run_times.csv")):
        Gt = pd.read_csv(p("resources_from_log", "gnn_run_times.csv")); Gt["spe"] = Gt.secs / Gt.epochs
        res_g = Gt.groupby("arm").spe.mean().to_dict()
        Xt = pd.read_csv(p("resources_from_log", "xgb_fit_times.csv")); res_x = Xt.groupby("arm").fit_s.mean().to_dict()
        rsrc = "taken from the run logs"
    else:
        rsrc = None
    nums["res_gnn_s_per_epoch"] = res_g; nums["res_xgb_fit_s"] = res_x
    rows = []
    for arm in ["G0_none", "G2_distance", "G4_static_wind", "G5_hourly_wind", "G6_vent_gated", "G7_adaptive"]:
        r = agg[agg.arm == arm]
        if r.empty: continue
        cells = []
        for h in HS:
            q = r[r.h == h].iloc[0]; v = f"{q['mean']:.2f}"
            if abs(q["mean"] - best[h]) < 1e-9: v = r"\textbf{" + v + "}"
            cells.append(v)
        if rsrc:
            cells.append(f"{res_g[arm]:.1f}" if arm in res_g else "--")
            cells.append(f"{res_x[arm]:.1f}" if arm in res_x else "--")
        rows.append(NAMES[arm] + " & " + " & ".join(cells) + r" \\")
    extra_hdr = r" & s/ep. & XGB s" if rsrc else ""
    ncol = 1 + len(HS) + (2 if rsrc else 0)
    cap = (r"Robustness check with the Graph WaveNet-style network (test year %s): MAE averaged over seeds (standard deviation $\le$ %.2f)." % (gyear, agg["std"].max()))
    if rsrc: cap += r" Resource use on a T4 GPU (" + rsrc + r"): GNN seconds per training epoch and XGBoost seconds per fitted model."
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n" r"\caption{" + cap + "}" "\n"
           r"\label{tab:gnn}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{2.6pt}" "\n"
           r"\begin{tabular}{@{}l" + "c" * (ncol - 1) + r"@{}}" "\n" r"\toprule" "\n"
           r"Graph & 1 h & 3 h & 6 h & 12 h & 24 h" + extra_hdr + r" \\" "\n" r"\midrule" "\n"
           + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(HERE, "tables", "tab_gnn.tex"), "w").write(tex)

# ---------------- Figure: lag asymmetry ----------------
f = p("lag_deseason", "lag_test.json")
if os.path.exists(f):
    L = json.load(open(f)); nums["lag_deseason"] = L
    Lr = json.load(open(p("lag", "lag_test.json"))) if os.path.exists(p("lag", "lag_test.json")) else None
    if Lr: nums["lag_raw"] = Lr
    lags = sorted(int(k) for k in L["mean_down_minus_up_by_lag"])
    fig, ax = plt.subplots(figsize=(COLW, 1.35))
    ax.axhline(0, c="0.6", lw=0.6); ax.axvline(0, c="0.6", lw=0.6)
    ax.plot(lags, [L["mean_down_minus_up_by_lag"][str(k)] for k in lags], marker="o", ms=2.5, c="#1f3b57", label="daily cycle removed")
    if Lr: ax.plot(lags, [Lr["mean_down_minus_up_by_lag"][str(k)] for k in lags], ls="--", c="#d62728", label="raw changes")
    ax.set_xlabel(r"lag $\tau$ (h)"); ax.set_ylabel(r"$c_{down}-c_{up}$"); ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_lag.pdf")

# ---------------- Figure: % change vs no graph, by horizon (both years) ----------------
yrs = YEARS
if yrs:
    fig, axs = plt.subplots(1, len(yrs), figsize=(COLW, 1.75), sharey=True)
    axs = np.atleast_1d(axs)
    arms = ["G1_uniform", "G2_distance", "G3_correlation", "G4_static_wind", "G5_hourly_wind", "G6_vent_gated"]
    for ax, y in zip(axs, yrs):
        O = pd.read_csv(p(f"stats_test{y}", f"xgb{y}_overall.csv")); HS = sorted(O.h.unique()); w = 0.8 / len(arms)
        for k, a in enumerate(arms):
            vals = [O[(O.h == h) & (O.model == a)].pct_vs_G0.values[0] for h in HS]
            ax.bar(np.arange(len(HS)) + (k - (len(arms) - 1) / 2) * w, vals, w, color=COL[a], label=NAMES[a])
        ax.axhline(0, c="k", lw=0.5); ax.set_xticks(range(len(HS))); ax.set_xticklabels([f"{h}" for h in HS])
        ax.set_xlabel("horizon (h)"); ax.set_title(f"test {y}")
    axs[0].set_ylabel("MAE change vs. no graph (%)")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, fontsize=5.6, bbox_to_anchor=(0.55, 1.13))
    save(fig, "fig_comparison.pdf")

# ---------------- Figure: wind vs distance across transport-ratio bins ----------------
if regs:
    fig, axs = plt.subplots(1, len(regs), figsize=(COLW, 1.4), sharey=True); axs = np.atleast_1d(axs)
    for ax, (y, d) in zip(axs, sorted(regs.items(), reverse=True)):
        sub = d[d.regime == "rho_bin"]
        for lev, lab, c in [("<0.5", r"$\rho<0.5$", "#9ecae1"), ("0.5-2", r"$0.5\leq\rho\leq2$", "#1f5fa8"),
                            (">2", r"$\rho>2$", "#08306b"), ("no_upwind", "no upwind", "#999999")]:
            q = sub[(sub.level == lev) & (sub.n >= 500)].sort_values("h")
            if not q.empty: ax.plot(q.h, q.pct_G5_vs_G2, marker="o", ms=2.5, c=c, label=lab)
        ax.axhline(0, c="k", lw=0.5); ax.set_xscale("log"); ax.set_xticks([1, 3, 6, 12, 24]); ax.set_xticklabels(["1", "3", "6", "12", "24"])
        ax.set_xlabel("horizon (h)"); ax.set_title(f"test {y}")
    axs[0].set_ylabel("hourly wind vs. distance (%)")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, fontsize=5.6, bbox_to_anchor=(0.55, 1.1))
    save(fig, "fig_rho.pdf")

# ---------------- Figures: training dynamics + resources (step 9) ----------------
if RES and os.path.exists(os.path.join(RES, "gnn_curves.json")):
    gc = json.load(open(os.path.join(RES, "gnn_curves.json"))); xc = json.load(open(os.path.join(RES, "xgb_curves_h6.json")))
    fig, axs = plt.subplots(1, 2, figsize=(COLW, 1.55))
    for a, c in gc.items(): axs[0].plot(range(1, len(c["val_mae"]) + 1), c["val_mae"], c=COL.get(a, "k"), label=NAMES[a])
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("val. MAE"); axs[0].set_title("GNN")
    for a, c in xc.items(): axs[1].plot(c["val"], c=COL.get(a, "k"), lw=0.8, label=NAMES[a])
    axs[1].set_xlabel("boosting round"); axs[1].set_title("XGBoost, 6 h"); axs[1].set_xscale("log")
    axs[0].legend(frameon=False, fontsize=4.8)
    save(fig, "fig_training.pdf")
    Xr = pd.read_csv(os.path.join(RES, "xgb_resources_h6.csv")); Gr = pd.read_csv(os.path.join(RES, "gnn_resources.csv"))
    nums["res_xgb"] = Xr.to_dict("records"); nums["res_gnn"] = Gr.to_dict("records")
    fig, axs = plt.subplots(1, 2, figsize=(COLW, 1.6))
    axs[0].barh([NAMES[a] for a in Xr.arm], Xr.graph_build_s, color="#ff7f0e", label="graph build")
    axs[0].barh([NAMES[a] for a in Xr.arm], Xr.fit_s, left=Xr.graph_build_s, color="#1f77b4", label="fit")
    axs[0].set_xlabel("XGBoost time (s)"); axs[0].legend(frameon=False, fontsize=5)
    axs[1].barh([NAMES[a] for a in Gr.arm], Gr.infer_ms_per_batch512, color="#2ca02c")
    axs[1].set_xlabel("GNN inference, 512 origins (ms)")
    save(fig, "fig_resources.pdf")
    rows = []
    for q in Gr.itertuples():
        x = Xr[Xr.arm == q.arm]
        xs = f"{x.graph_build_s.values[0]:.1f} / {x.fit_s.values[0]:.0f}" if len(x) else "--"
        rows.append(f"{NAMES[q.arm]} & {xs} & {q.params/1e3:.1f} & {q.mean_epoch_s:.1f} & {q.peak_gpu_mb:.0f} & {q.infer_ms_per_batch512:.1f} \\\\")
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Resource use on a T4 GPU: XGBoost graph-build / fit time (s, 6~h horizon); GNN parameters ($\times10^3$), time per epoch (s), peak GPU memory (MB) and inference time for 512 forecast origins (ms).}" "\n"
           r"\label{tab:resources}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
           r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n" r"Graph & XGB build/fit & Params & s/epoch & GPU MB & Inf. ms \\" "\n" r"\midrule" "\n"
           + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    open(os.path.join(HERE, "tables", "tab_resources.tex"), "w").write(tex)

# misc numbers
for k in ["prep/stations.csv"]:
    if os.path.exists(p(k)): nums["n_stations"] = int(len(pd.read_csv(p(k))))
for y in YEARS:
    f = p(f"xgb_test{y}", "tuning.json")
    if os.path.exists(f): nums[f"tuning_{y}"] = json.load(open(f))
    f = p(f"stats_test{y}", f"xgb{y}_H3_conditional.csv")
    if os.path.exists(f): nums[f"H3_{y}"] = pd.read_csv(f).to_dict("records")
json.dump(nums, open(os.path.join(HERE, "results_numbers.json"), "w"), indent=1, default=float)
print("tables:", os.listdir(os.path.join(HERE, "tables"))); print("figures:", os.listdir(os.path.join(HERE, "figures")))
