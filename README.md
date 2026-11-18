# Do Wind-Directed Graphs Add Information Beyond Station History?

Code and results for the paper *"Do Wind-Directed Graphs Add Information Beyond Station History? A Transport-Conditioned Evaluation on Delhi's Air-Quality Network"* by Raushan Kumar, Priya Sahoo and Grandhi Saketh (SRM University–AP).

The study asks when, if ever, a wind-directed station graph improves multi-station PM2.5 forecasts in Delhi. The learner is held fixed and only the graph changes. Wind graphs are compared with no graph, uniform, distance and correlation graphs, after a model-free test of whether a wind-transport signal exists between stations at all.

## Run it

1. Open `notebook/delhi_windgraph_colab.ipynb` in Google Colab.
2. Choose Runtime → Change runtime type → **T4 GPU**.
3. Choose Runtime → **Run all**.

The notebook installs its requirements, downloads the data, writes the scripts in `src/`, runs every step and downloads the results. No API keys are needed.

## Structure

```
notebook/delhi_windgraph_colab.ipynb   full pipeline for Google Colab
src/download_data.py      Delhi rows of the CPCB archive
src/prepare_data.py       hourly PM2.5 panel, quality control, ERA5 wind and boundary-layer height
src/lag_test.py           model-free upwind/downwind lag test
src/graph_experiment.py   7 graph constructions x 5 horizons, fixed XGBoost learner
src/evaluate_stats.py     MAE/RMSE/R2, Diebold-Mariano + Holm, week-block bootstrap, regime breakdowns
src/gnn_experiment.py     Graph WaveNet-style robustness check
src/resources.py          training curves, run time, GPU memory, inference latency
tools/make_paper_tables.py  builds the paper's LaTeX tables and figures from the results
diagrams/                 draw.io sources of the block and flow diagrams
data/station_coords.csv   station latitude/longitude
results/                  summary results of the reported run (CSV/JSON)
```

## Reported results (`results/`)

From the Colab run reported in the paper (34 stations, 2019–2024; test years 2023 and 2024; Tesla T4, Python 3.13, PyTorch 2.14, XGBoost 3.4):

- **Transport test:** downwind flow raises the lagged correlation of hourly PM2.5 changes in 98.6% of 514 station pairs (Wilcoxon p < 1e-80). The effect peaks at a 1 h lag and fades by about 5 h.
- **Graph vs. no graph:** neighbour information lowers MAE by 5–9% at 1–3 h in 2024 and 3–5% in 2023, with no gain at 12–24 h.
- **Wind vs. distance:** the hourly wind graph is never significantly better than an undirected distance graph, and is 1.3–1.9% worse at 3–12 h in 2024.
- **GNN check:** the same pattern holds, and the learned adaptive graph is best.

`results/test*/xgb*_overall.csv` were produced by `src/evaluate_stats.py` from the Colab prediction files. `results/run_times/` holds training times parsed from the run logs.

## Data sources

- **PM2.5:** Central Pollution Control Board monitoring data, via [Vonter/india-cpcb-aqi](https://github.com/Vonter/india-cpcb-aqi) (ODbL-1.0). Timestamps are treated as local time (IST).
- **Meteorology:** ERA5 reanalysis, retrieved through the [Open-Meteo](https://open-meteo.com) historical API.
- **Station coordinates:** XKDR air-quality station catalogue, as distributed in [sharifmdathar/delhi-ncr-pm25](https://github.com/sharifmdathar/delhi-ncr-pm25) (MIT).

Raw data are downloaded by the notebook and are not stored here.

## Setup

- **Splits:** rolling test years Y: train on 2019 to Y−2, validate on Y−1, test on Y. The notebook supports 2023–2025.
- **Horizons:** 1, 3, 6, 12 and 24 h.
- **No future weather:** meteorology is used only at the forecast origin.
- **Common starting point:** every model predicts the change from the station's current value.
