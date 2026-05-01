# 📊 Canadian Retail Sales Forecasting — Time Series Analysis

> **Time series forecasting project** using a Statistics Canada-structured retail trade dataset spanning 10 years (2015–2024) across 10 sectors. Built a from-scratch AR(13) forecasting model with seasonal differencing, COVID-19 anomaly detection, and sector-by-sector recovery analysis — no external forecasting libraries used.

---

## 📌 Project Overview

Canada's retail sector generates over **$630 billion CAD annually** across 10 distinct trade categories. This project applies classical time series techniques to decompose, analyze, and forecast monthly retail sales — the exact kind of analysis used by banks, consulting firms, and government agencies for economic planning.

**The analytical challenge:** Retail data contains overlapping signals — long-term trend, monthly seasonality, a massive COVID-19 structural break, and an inflation distortion layer in 2021–2023. Untangling these components correctly is what separates real analysis from surface-level dashboarding.

---

## 📊 Key Findings

| Finding | Detail |
|---|---|
| Dataset | 1,200 rows × 10 sectors × 120 months (2015–2024) |
| Market size | $632B CAD annually (2024) |
| 9-year nominal growth | +21.4% |
| Hardest COVID-hit sector | Clothing & Accessories (−71.8% in Dec 2020) |
| Most resilient sector | Food & Beverage (< −5% impact) |
| Fastest COVID recovery | Electronics, Food & Beverage (3 months) |
| Slowest recovery | Clothing & Accessories (14 months) |
| 2025 Total Market Forecast | **$641B CAD** (+1.4% vs 2024) |
| Best model fit (MAE%) | Food & Beverage — **3.2%** |
| Avg model MAE% | 9.5% across all 10 sectors |

---

## 🗂️ Repository Structure

```
canadian-retail-forecasting/
│
├── data/
│   └── retail_trade.csv           # 1,200-row dataset (10 sectors × 120 months)
│
├── outputs/
│   ├── figure1_decomposition.png  # STL-style decomposition: trend/seasonal/residual
│   ├── figure2_covid_impact.png   # COVID heatmap + recovery timeline
│   ├── figure3_forecast.png       # 12-month forecasts per sector + total market
│   └── figure4_seasonal_market.png # Seasonal indices, market share, YoY growth
│
├── canadian_retail_forecasting.py  # Full analysis script
└── README.md
```

---

## 🔍 Analysis Breakdown

### 1. Exploratory Data Analysis
- Loaded and profiled 10 retail sectors across 10 years
- Computed descriptive statistics: mean, min, max, standard deviation per sector
- Calculated annual totals and 9-year compound growth rate

### 2. Classical Time Series Decomposition (Additive)
Manual implementation of STL-style decomposition:

```
Y(t) = Trend(t) + Seasonal(t) + Residual(t)
```

- **Trend**: 12-month centred moving average
- **Seasonal**: Average detrended values by calendar month (centred to zero)
- **Residual**: Unexplained variation after removing trend and seasonality
- **Seasonal Strength metric**: `1 - Var(Residual) / Var(Detrended)` (range: 0–1)

| Sector | Seasonal Strength | Peak Month | Trough Month |
|---|---|---|---|
| Food & Beverage | 0.668 (Strong) | May | November |
| Clothing | 0.595 (Strong) | June | December |
| Motor Vehicles | 0.564 (Strong) | June | December |

### 3. COVID-19 Anomaly Detection
- Baseline: 2015–2019 seasonal averages per sector-month
- Anomaly score: % deviation from baseline for each month in 2020–2021
- Impact categorization: Essential / Moderate / Hard Hit / Severely Disrupted

```python
# COVID impact calculation
pct_change = (actual - baseline) / baseline * 100
```

### 4. Recovery Timeline Analysis
Defined recovery as: first month where sales returned within ±5% of pre-COVID baseline

| Sector | Recovery Time | Recovery Date |
|---|---|---|
| Electronics & Appliances | 3 months | Jun 2020 |
| Food & Beverage Stores | 3 months | Jun 2020 |
| Clothing & Accessories | 14 months | May 2021 |

### 5. AR(13) Forecasting Model (Built from Scratch)
No forecasting library used — model implemented with numpy OLS:

**Steps:**
1. Apply **seasonal differencing** (lag-12): `d_t = y_t − y_{t−12}`
2. Build AR(13) design matrix from lagged differenced values
3. Fit coefficients using **Ordinary Least Squares**: `β = (X'X)⁻¹X'Y`
4. Forecast iteratively for 12 steps ahead
5. Invert seasonal differencing to recover original scale
6. Generate **95% confidence intervals** (expanding with forecast horizon)

```python
# OLS solution
beta = np.linalg.solve(XtX + np.eye(len(XtX)) * 1e-6, XtY)
```

**Why AR(13)?** With monthly data, lag-13 captures both the immediate AR(1) effect and the full seasonal lag (12) plus one step — empirically the best-performing single-lag specification for this data.

### 6. Model Evaluation
| Metric | Description |
|---|---|
| MAE | Mean Absolute Error (in CAD millions) |
| RMSE | Root Mean Squared Error (in CAD millions) |
| MAE% | MAE as % of mean — allows cross-sector comparison |

---

## 📈 Visualizations

### Figure 1 — Time Series Decomposition
Three key sectors decomposed into original series, trend, seasonal, and residual components. COVID shock period shaded in red. Residual outliers flagged with ⚠.

### Figure 2 — COVID-19 Impact Heatmap
Annotated heatmap showing % deviation from pre-COVID baseline for all 10 sectors across Jan 2020–Dec 2021. Right panel: horizontal bar chart of recovery timelines.

### Figure 3 — 12-Month Sector Forecasts (2025)
Individual forecast panels for all 10 sectors + total market. Shows last 3 years of actuals, forecast line with 95% CI shading, and in-panel MAE% score.

### Figure 4 — Seasonal Patterns & Market Structure
Seasonal index curves by sector, market share pie, year-over-year growth time series, and annual total retail bar chart with 2025 forecast.

---

## 💡 Key Insights

1. **Essential vs Discretionary divide** is statistically measurable — Food/Health showed near-zero COVID impact while Clothing/Gasoline fell 60–70%

2. **Inflation distortion (2021–2023)**: Nominal sales rose 6–9% but real volume growth was lower — critical caveat for any revenue-only analysis

3. **Seasonal structure is consistent** across years — Motor Vehicles peaks May–June (tax refund + spring), retail universally weakest in January

4. **Sporting Goods post-COVID boom** recovered faster than pre-COVID trend due to outdoor/home fitness surge — a structural demand shift, not a bounce-back

5. **2025 outlook**: $641B CAD forecast (+1.4%) reflects post-inflation normalization moderating growth to historical 2–3% real rate

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.12` | Core analysis |
| `pandas` | Data manipulation |
| `numpy` | AR model (OLS), decomposition, numerical operations |
| `scipy.stats` | Statistical testing |
| `matplotlib` | All visualizations |
| `seaborn` | Heatmap (COVID impact) |

> **Note:** This project intentionally implements ARIMA-style forecasting from first principles using numpy rather than relying on `statsmodels` — demonstrating understanding of the underlying mathematics, not just library usage.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/canadian-retail-forecasting.git
cd canadian-retail-forecasting

# Install dependencies (minimal)
pip install pandas numpy matplotlib seaborn scipy

# Run the full analysis
python canadian_retail_forecasting.py
```

All 4 figures will be saved to the `outputs/` folder. Estimated runtime: ~30 seconds.

---

## 📝 Data Notes

This project uses a **simulated dataset** built to match the structure, scale, and known patterns of Statistics Canada Table 20-10-0063-01 (Retail trade, sales by province and territory). The simulation incorporates:
- Real sector-level average sales magnitudes (CAD millions)
- Documented COVID-19 impact patterns by retail category
- Known inflation dynamics from 2021–2023
- Statistics Canada-aligned seasonal patterns

To use the actual Statistics Canada data, download Table 20-10-0063-01 from [www150.statcan.gc.ca](https://www150.statcan.gc.ca) and replace `data/retail_trade.csv` with the downloaded file, adjusting column names as needed.

---

## 👤 About This Project

Built as part of a data analytics portfolio to demonstrate real-world time series skills — decomposition, anomaly detection, from-scratch forecasting, and clear business communication.

**Analyst:** [Your Name] | [LinkedIn] | [Portfolio]

---

*Inspired by: Statistics Canada Table 20-10-0063-01 — Retail trade, sales by province and territory*
