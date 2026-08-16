# Project 01 — AI-Powered BESS Optimization

## 1. Project Overview

Build a production-quality portfolio project that combines machine learning forecasting with mathematical optimization for a grid-connected Battery Energy Storage System (BESS).

The system should forecast:
- Electricity demand
- Renewable generation (preferably PV)
- Electricity price

Then use those forecasts inside a mathematical optimization model to determine:
- Battery charging
- Battery discharging
- Grid import
- Grid export
- State of charge (SOC)

The primary objective is to minimize total electricity cost while respecting battery, grid, and operational constraints.

This project is intended as a professional portfolio piece for remote/freelance work in:
- Energy optimization
- Python optimization
- Pyomo
- Operations research
- Energy analytics
- Machine learning for energy systems

## 2. Core Story

The repository must communicate this workflow:

Historical data
→ Data validation and preprocessing
→ Forecasting
→ Optimization
→ Dispatch schedule
→ Economic evaluation
→ Visualization
→ Scenario analysis
→ Reproducible report

Do not build a generic ML notebook. The forecasting model exists because its output drives an optimization decision.

## 3. Target Technical Stack

Primary:
- Python 3.11+
- pandas
- NumPy
- scikit-learn
- XGBoost or LightGBM where appropriate
- Pyomo
- HiGHS as the default open-source MILP solver
- matplotlib
- Plotly
- Streamlit for an optional dashboard
- pytest
- Jupyter only for exploratory analysis

Optional:
- Gurobi for benchmarking
- MLflow or a lightweight experiment tracker
- Pydantic for configuration/data validation

Use a solver abstraction so the project is not hard-coded to one commercial solver.

## 4. Repository Structure

Create approximately:

```text
ai-bess-optimization/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── base.yaml
│   └── scenarios/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── forecasting/
│   ├── optimization/
│   ├── evaluation/
│   └── visualization/
├── tests/
├── reports/
├── results/
└── app/
```

Keep business logic out of notebooks.

## 5. Phase 0 — Requirements Definition

Define a reference case before coding.

Recommended baseline:
- 24-hour or 7-day chronological horizon
- 15-minute or hourly resolution
- One behind-the-meter load
- One PV plant
- One BESS
- Time-varying electricity tariff
- Optional export compensation

Recommended initial BESS:
- Energy capacity: 1 MWh
- Maximum charge power: 0.5 MW
- Maximum discharge power: 0.5 MW
- Minimum SOC: 10%
- Maximum SOC: 90%
- Charge efficiency: 95%
- Discharge efficiency: 95%
- Initial SOC: 50%

These are simulation assumptions, not real equipment claims.

## 6. Phase 1 — Data Strategy

Use a public dataset where possible. If public data is unavailable or inconvenient, generate a synthetic dataset from realistic profiles, clearly labeling it as synthetic.

Required time series:
- Load
- Solar generation or irradiance
- Electricity price
- Timestamp

Optional:
- Temperature
- Cloud cover
- Wind
- Calendar features

Data requirements:
- Chronological timestamps
- Explicit timezone handling
- No leakage from future observations
- Documented units
- Missing-value strategy
- Outlier strategy

Create a data dictionary.

## 7. Phase 2 — Exploratory Data Analysis

Analyze:
- Daily load profile
- PV profile
- Price profile
- Seasonal patterns
- Correlation between variables
- Missing values
- Outliers
- Peak demand
- Net load

Produce professional figures:
1. Load profile
2. PV generation
3. Electricity price
4. Net load
5. Representative daily profiles
6. Correlation matrix
7. Distribution plots

Do not overproduce charts. Every chart must answer a question.

## 8. Phase 3 — Forecasting

Build baseline models first.

Load forecasting:
- Persistence baseline
- Linear regression
- Random Forest or Gradient Boosting
- XGBoost if justified

PV forecasting:
- Persistence baseline
- Regression/tree model
- XGBoost if features support it

Price forecasting:
- Persistence or rolling baseline
- Tree-based model

Use time-series validation, not random train/test splitting.

Recommended evaluation:
- MAE
- RMSE
- MAPE where mathematically appropriate
- sMAPE where MAPE is problematic
- R² only as a supplementary metric

Report:
- Baseline performance
- ML performance
- Error by hour
- Error during peak periods

## 9. Phase 4 — Forecast-to-Optimization Interface

Create a clean interface:

```text
ForecastResult
    timestamps
    load_forecast
    pv_forecast
    price_forecast
    prediction_metadata
```

The optimizer must consume forecasts through a defined interface rather than directly reading notebook variables.

This is a critical software-engineering requirement.

## 10. Phase 5 — Mathematical Optimization Model

Formulate a linear or mixed-integer optimization model in Pyomo.

### Decision variables

At minimum:
- Grid import
- Grid export
- Battery charge power
- Battery discharge power
- SOC

Optional binary:
- Battery charging state
- Battery discharging state

### Power balance

For each time step:

```text
Load + BatteryCharge + GridExport
=
PV + BatteryDischarge + GridImport
```

Adapt the equation to the exact sign convention used in the implementation.

### SOC dynamics

Use:

```text
SOC[t+1] =
SOC[t]
+ η_charge * Charge[t] * Δt
- Discharge[t] * Δt / η_discharge
```

### Constraints

Implement:
- SOC lower bound
- SOC upper bound
- Charge power limit
- Discharge power limit
- Initial SOC
- Optional terminal SOC
- Grid import limit
- Grid export limit
- Optional no-simultaneous-charge/discharge constraint

Avoid unnecessary binary variables if the formulation remains valid without them.

## 11. Objective Function

Primary objective:

```text
Minimize:
energy purchase cost
- export revenue
+ optional battery degradation cost
+ optional demand charge
```

Run three versions:
1. Cost-only
2. Cost + degradation
3. Cost + degradation + demand charge

Document every coefficient.

## 12. Phase 6 — Battery Degradation

Implement a transparent simplified degradation model first.

Possible approach:
- Throughput-based degradation cost

Example concept:

```text
degradation_cost =
battery_throughput × degradation_cost_per_MWh
```

Do not claim electrochemical accuracy.

Clearly state that this is an economic proxy.

Optional advanced extension:
- Piecewise-linear degradation
- Cycle-depth approximation
- Rainflow analysis as post-processing

Do not make rainflow optimization unnecessarily complex in version 1.

## 13. Phase 7 — Perfect-Foresight Benchmark

Before using ML forecasts, solve the optimization with actual future data.

This creates an upper-bound/reference solution:

```text
Actual data → Perfect foresight optimizer
Forecast data → Forecast-based optimizer
```

Compare:
- Cost
- Peak demand
- Battery throughput
- Renewable curtailment
- Grid import
- Grid export

This is essential for evaluating the value of forecasting.

## 14. Phase 8 — Rolling-Horizon Simulation

Implement a realistic rolling-horizon controller.

At each decision time:
1. Collect information available at that time.
2. Generate forecasts.
3. Solve the optimization problem.
4. Implement only the first control interval.
5. Advance time.
6. Repeat.

Never allow future actual data to enter the forecast at the current time.

Compare:
- Perfect foresight
- Day-ahead forecast
- Rolling forecast
- No-battery baseline

## 15. Phase 9 — Evaluation

Report:
- Total electricity cost
- Cost savings %
- Peak demand
- Peak reduction %
- Battery throughput
- Average SOC
- Renewable self-consumption
- Renewable curtailment
- Grid import
- Grid export
- Forecast errors
- Solver runtime

Create a baseline table.

## 16. Phase 10 — Scenario Analysis

At minimum:
1. No battery
2. Battery without optimization
3. Optimized battery
4. Forecast-based optimization
5. Perfect-foresight optimization
6. Higher electricity prices
7. Lower PV generation
8. Higher load
9. Different battery capacities

Generate a sensitivity analysis for:
- Battery energy capacity
- Battery power rating
- Round-trip efficiency
- Degradation cost
- Price volatility

## 17. Phase 11 — Dashboard

Optional Streamlit dashboard.

Sections:
- System configuration
- Forecasts
- Optimal dispatch
- SOC
- Cost comparison
- KPI summary
- Scenario controls

Dashboard should be a visualization layer, not the optimization engine.

## 18. Testing

Write tests for:
- Data validation
- SOC equation
- Power balance
- Constraint bounds
- Objective calculation
- Forecast leakage
- Optimization feasibility
- Result serialization

Include at least one end-to-end test using a small synthetic dataset.

## 19. Reproducibility

Provide:
- Environment specification
- Installation instructions
- Data acquisition instructions
- Configuration files
- Fixed random seeds where appropriate
- One command to reproduce the baseline experiment

## 20. README Requirements

README must contain:
1. Problem statement
2. Why it matters
3. Architecture
4. Mathematical formulation
5. Data source
6. Forecasting methodology
7. Optimization methodology
8. Results
9. Limitations
10. How to run
11. Future work

Include an architecture diagram.

## 21. Portfolio Quality Requirements

The final repository must look like professional engineering work, not a university assignment.

Prioritize:
- Clean architecture
- Reproducibility
- Explainable assumptions
- Strong README
- Meaningful visualizations
- Tests
- Config-driven experiments
- Clear limitations

## 22. Stretch Goals

After the baseline is complete:
- Probabilistic forecasting
- Chance-constrained optimization
- Stochastic programming
- Distributionally robust optimization
- Real-time price API integration
- BESS degradation model
- Grid services
- Frequency regulation revenue
- Co-optimization of PV, BESS and EVs

Do not start stretch goals until the baseline is complete.

## 23. Definition of Done

The project is complete when:
- A new user can clone and run it.
- Forecasts are generated without leakage.
- Pyomo optimization solves successfully.
- Physical constraints are respected.
- Rolling-horizon simulation works.
- Baseline comparisons are reproducible.
- Results are documented.
- Tests pass.
- README explains the project professionally.

---

# Decisions made after this brief

The text above is the brief as written, preserved unchanged. This section
records every decision taken after it — including the places where the brief
was overridden, and why. Nothing here was decided silently.

## A. Repository name (supersedes §4)

The brief names the repo `ai-bess-optimization`. It is called
**`forecast-driven-bess-dispatch`** instead.

There is a sibling repo, `battery-storage-optimization-pyomo`, which is a
*wholesale-market* battery model: given a known price series, co-optimize
energy arbitrage against frequency-regulation capacity. Two repos named
`ai-bess-optimization` and `battery-storage-optimization-pyomo` sitting side by
side read as two attempts at one idea. The new name states the actual
distinction: here the prices are **not known**, and the whole point is that
dispatch is decided against forecasts.

The Python package is `bess_dispatch`, not `bess_opt`, so both repos can be
installed into one environment without colliding.

## B. Data: real, not synthetic (§6)

§6 permits synthetic data. It is used only as a fallback, because synthetic data
would make the project's central result circular: the gap between
forecast-driven and perfect-foresight dispatch would be a measurement of
whatever noise the generator injected, not of forecasting difficulty.

Primary data is the **Open Power System Data** hourly time series, pinned at the
`2020-10-06` snapshot. A seeded synthetic generator ships alongside for offline
use and CI. See `data/DATA_DICTIONARY.md` and `data/ATTRIBUTION.md`.

Consequence worth stating plainly: the data is a **national aggregate rescaled
to one site**. The shapes are real; the magnitudes are site assumptions. A
national load curve is smoother, and therefore easier to forecast, than a single
building's. Rather than hide that, the evaluation includes an ablation isolating
which of the three forecasts the money actually depends on.

## C. Data window is 2018-10-01 to 2020-09-30 (§6)

Not an arbitrary truncation. The German/Austrian bidding zone split on
2018-10-01, so a `DE_LU` price series does not exist before that date.
Concatenating the earlier `DE_AT_LU` series onto it would silently join two
different markets, so it is not done.

## D. No binary variables (§10)

§10 asks to avoid unnecessary binaries and lists no-simultaneous-charge/discharge
as optional. Probe runs on real data confirmed the optimum never charges and
discharges in the same period — across export compensation ratios 0.7x, 1.0x and
1.3x, with and without degradation cost, with and without a demand charge.
Round-trip efficiency losses already make it strictly wasteful. The model is a
pure LP.

What the probe *did* find is a different failure mode, and it is a data problem
rather than a model one: when export compensation exceeds the import tariff, the
optimum imports and exports simultaneously in every hour — **with no battery in
the model at all**. That is the meter being gamed, not the battery. The guard
therefore lives in schema validation (`export_price <= import_price` per period),
not in a binary variable. Adding binaries would have buried a data error behind
a slower solve.

## E. The reference battery is kept, and its weak result reported (§5)

§5's recommended 1 MWh / 0.5 MW battery saves about **3.5%** of energy cost on
this site — measured, not estimated. Larger batteries do better (2 MWh / 1 MW:
~6.7%; 4 MWh / 2 MW: ~11.7%).

The brief's sizing is kept as the documented reference case rather than quietly
replaced with a more flattering one. The README leads with the sizing curve, so
the honest answer — value scales with sizing, and 3.5% is what the brief's own
reference case is worth — is the headline.

## F. Plotly is skipped (§3)

§3 lists both matplotlib and Plotly. Only matplotlib is used. The committed
figures must render headless in CI as PNGs, and the Streamlit dashboard does not
need Plotly to do its job. Listing a dependency the project does not exercise
would be worse than omitting it.

## G. Package layout (§4)

§4's tree puts `data/`, `forecasting/`, `optimization/` directly under `src/`.
They are nested under `src/bess_dispatch/` instead, so the project is a real
installable package rather than a loose directory of modules. Everything else in
§4's tree is preserved.

## H. XGBoost is optional and conditional (§3, §8)

§3 says "XGBoost or LightGBM where appropriate" and §8 says "XGBoost if
justified". It is behind an optional `boost` extra and is reported on only if it
measurably beats scikit-learn's `HistGradientBoostingRegressor` on the held-out
window. Whether it did is recorded in the docs rather than assumed here.

## I. Build sequencing (§23)

Built phase by phase with a commit each, verified locally, then pushed as a
complete history — matching the four sibling repos.
