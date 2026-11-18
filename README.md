# Do Wind-Directed Graphs Add Information Beyond Station History?

**A transport-conditioned evaluation on Delhi's air-quality monitoring network**

Raushan Kumar, Priya Sahoo, Grandhi Saketh
Department of Computer Science and Engineering, SRM University–AP, Amaravati, India

---

## Overview

Many recent air-quality forecasting models connect monitoring stations with graph edges that follow the wind, on the idea that pollution moves from upwind stations to downwind ones. This project tests whether that idea actually improves forecasts.

We forecast hourly PM2.5 at 34 stations in Delhi and compare seven ways of building the station graph, keeping everything else fixed. Before training any model, we also check directly whether pollution moves between stations with the wind.

**Main findings**

1. Pollution does travel between Delhi stations with the wind, but only over one to three hours.
2. Using information from neighbouring stations lowers short-range forecast error by 3–9%.
3. Pointing the graph edges in the wind direction does not help: an ordinary distance-based graph does as well or better.

The accompanying paper (IEEE conference format, 6 pages) reports these results in full.

---

## Quick start (Google Colab)

The whole study runs from one notebook. No uploads or API keys are needed.

1. Open `notebook/delhi_windgraph_colab.ipynb` in [Google Colab](https://colab.research.google.com).
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Select **Runtime > Run all**.

The notebook installs the requirements, downloads the data, runs all experiments and downloads the results as a zip file. A full run takes a few hours on a free T4 GPU.

---

## Repository structure

| Path | Contents |
|---|---|
| `notebook/delhi_windgraph_colab.ipynb` | Complete pipeline for Google Colab |
| `src/download_data.py` | Downloads the Delhi rows of the CPCB monitoring archive |
| `src/prepare_data.py` | Builds the hourly PM2.5 panel, applies quality control, fetches ERA5 weather data |
| `src/lag_test.py` | Model-free test of wind-driven transport between stations |
| `src/graph_experiment.py` | Main experiment: seven graph types, five forecast horizons, XGBoost |
| `src/evaluate_stats.py` | Error metrics, significance tests and breakdowns by season, ventilation and transport |
| `src/gnn_experiment.py` | Robustness check with a graph neural network |
| `src/resources.py` | Training curves, run time and GPU memory measurements |
| `tools/make_paper_tables.py` | Turns the notebook's results zip into the tables and figures of the paper |
| `diagrams/` | draw.io source files of the block diagram and flow diagram, their PDF versions (`pdf/`) and the scripts that create them |
| `data/station_coords.csv` | Latitude and longitude of each station |
| `results/` | Summary results of the run reported in the paper |
| `requirements.txt` | Python packages |

---

## Method in brief

**Data**
- 34 Delhi monitoring stations, hourly PM2.5, 2019–2025.
- Weather (wind and boundary-layer height) from ERA5, used only at the moment a forecast is made, so no future information enters the models.

**Graph types compared**

| Name | How stations are connected |
|---|---|
| No graph | Each station uses only its own history |
| Uniform | Equal weight to all other stations |
| Distance | Closer stations get more weight |
| Correlation | The five stations most correlated in the training years |
| Static wind | Edges follow the average wind of the training years |
| Hourly wind | Edges follow the wind at the time of the forecast |
| Ventilation-gated | Switches between wind and distance edges depending on how well the air is mixing |

**Evaluation**
- Forecast horizons: 1, 3, 6, 12 and 24 hours.
- Three test years: each model is trained on earlier years, tuned on the year before the test year, and tested on the test year (2023, 2024 and 2025).
- Metrics: mean absolute error (MAE), root-mean-square error (RMSE) and R².
- Significance: Diebold–Mariano tests with Holm correction, and bootstrap confidence intervals.

---

## Results

Results of the reported run are in `results/`. Experiments ran on a Tesla T4 GPU with Python 3.13, PyTorch 2.14 and XGBoost 3.4.

| Question | Answer |
|---|---|
| Does pollution move between stations with the wind? | Yes. In 98.4% of 514 station pairs, changes travel downwind. The effect is strongest after 1 hour and gone after about 5 hours. |
| Does using neighbouring stations help? | Yes, at short horizons: 4–6% lower MAE at 1–3 hours in 2025, 5–9% in 2024 and 3–5% in 2023. No gain at 12–24 hours. |
| Does following the wind direction help? | No. Across 15 year and horizon combinations, the hourly wind graph is significantly worse than the distance graph in 8 and better in only 1 (by 0.25%). |
| Does a neural network change the picture? | No. The same pattern holds, and a graph learned from the data performs best. |

**Forecast error in the most recent test year (2025)**

Mean absolute error in µg/m³, XGBoost learner, 34 stations. Lower is better.

| Model | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| Persistence (last value) | 15.31 | 30.05 | 42.44 | 50.47 | 37.23 |
| No graph | 13.64 | 22.32 | 27.30 | 30.60 | 32.11 |
| Distance graph | 12.86 | 21.17 | 26.74 | 30.64 | 32.10 |
| Hourly wind graph | 12.91 | 21.34 | 26.85 | 30.56 | 32.21 |
| Ventilation-gated graph | 12.79 | 21.16 | 26.74 | 30.59 | 32.14 |

The full tables for all seven graphs and all three test years are in `results/test20XX/xgb20XX_overall.csv`.

**Files in `results/`**

| File | Contents |
|---|---|
| `test2025/`, `test2024/`, `test2023/` | Error metrics, significance tests and regime breakdowns for each test year |
| `gnn2025/`, `gnn2024/` | The same for the graph neural network |
| `resources/` | Training curves, run time, GPU memory and inference time for each graph |
| `lag_test/` | Results of the wind-transport test |
| `run_times/` | Training times taken from the run logs |
| `stations.csv` | Stations used and their data completeness |
| `env_info.json` | Hardware and software versions |

---

## Data sources

| Data | Source | Licence |
|---|---|---|
| PM2.5 measurements | Central Pollution Control Board, via the [Vonter/india-cpcb-aqi](https://github.com/Vonter/india-cpcb-aqi) archive | ODbL-1.0 |
| Weather | ERA5 reanalysis, via the [Open-Meteo](https://open-meteo.com) historical API | CC-BY 4.0 |
| Station coordinates | XKDR air-quality station catalogue, as distributed in [sharifmdathar/delhi-ncr-pm25](https://github.com/sharifmdathar/delhi-ncr-pm25) | MIT |

Raw data are downloaded by the notebook and are not stored in this repository. Timestamps in the CPCB archive are labelled UTC but are local time (IST); the code treats them as IST.

---

## Running locally

A CUDA GPU is recommended. With Python 3.10 or newer:

```bash
pip install -r requirements.txt
python src/download_data.py data 2019 2020 2021 2022 2023 2024 2025
python src/prepare_data.py data data/station_coords.csv out/prep 2025
python src/lag_test.py out/prep out/lag_deseason deseason
python src/graph_experiment.py out/prep out/xgb_test2025 2025
python src/evaluate_stats.py out/xgb_test2025/test_predictions.parquet out/xgb_test2025/met_origin.parquet out/prep/panel.npz out/stats_test2025 xgb2025
```

See the notebook for the remaining steps (other test years, GNN, resource measurements).

To rebuild the paper's tables and figures, unzip the `results.zip` that the notebook downloads and run:

```bash
python tools/make_paper_tables.py path/to/unzipped_results paper_tables
```

---

## Where each part of the paper comes from

| Paper | Produced by | Result files |
|---|---|---|
| Fig. 3, lag-asymmetry test | `src/lag_test.py` | `results/lag_test/` |
| Table II and Fig. 4, graph comparison | `src/graph_experiment.py`, `src/evaluate_stats.py` | `results/test2023/`, `test2024/`, `test2025/` (`*_overall.csv`) |
| Table III, direction ablation | `src/evaluate_stats.py` | `results/test20XX/*_pairwise.csv` |
| Table IV and Fig. 5, regime breakdown | `src/evaluate_stats.py` | `results/test20XX/*_regimes.csv`, `*_H3_conditional.csv` |
| Table V, neural network and run time | `src/gnn_experiment.py`, `src/resources.py` | `results/gnn2025/`, `results/resources/` |
| Figs. 1 and 2, block and flow diagrams | `diagrams/render_drawio.py` | `diagrams/pdf/` |

---

## Limitations

- One city and three test years.
- All stations fall inside four ERA5 grid cells, so the wind used is effectively the same across the city; station wind sensors were too unreliable to use.
- Much of Delhi's pollution arrives from outside the city, which no graph built on city stations can see.

---

## Citation

```bibtex
@misc{kumar2026windgraphs,
  title  = {Do Wind-Directed Graphs Add Information Beyond Station History? A Transport-Conditioned Evaluation on Delhi's Air-Quality Network},
  author = {Kumar, Raushan and Sahoo, Priya and Saketh, Grandhi},
  year   = {2026},
  note   = {Department of Computer Science and Engineering, SRM University--AP},
  url    = {https://github.com/raushan95a/wind-directed-graphs-delhi-aq}
}
```
