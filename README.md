# Real-Time Dynamic Hotel Room Pricing Engine
**Operations Research Techniques: Markov Decision Process (Dynamic Programming) & Network Revenue Management (Linear Programming)**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero External APIs](https://img.shields.io/badge/APIs-Zero%20External%20(Offline)-orange.svg)]()

A production-grade, offline-first real-time hotel room pricing engine designed using foundational **Operations Research (OR)** models. The engine dynamically optimizes room rates and inventory allocations as market conditions shift, check-in deadlines approach, and rooms are booked.

---

## 1. System Architecture & Key Modules

```
d:\11111\college\sem 3\OR project\
├── frontend/
│   ├── index.html                   # Single-page web dashboard (Bootstrap 5, Chart.js)
│   ├── app.js                       # Frontend controller, async fetch & reactive UI
│   └── style.css                    # Custom dashboard styling & KPI cards
├── server.py                        # Local FastAPI backend REST API & static web server
├── data/
│   └── market_conditions.csv        # Synthetic market feed dataset (lead time, comp prices, elasticity)
├── src/
│   ├── __init__.py                  # Package descriptor
│   ├── data_generator.py            # Generates realistic synthetic market feed (Zero APIs)
│   ├── demand_model.py              # Poisson arrival process & price elasticity model
│   ├── dp_solver.py                 # Finite-horizon MDP Bellman backward induction solver
│   ├── lp_solver.py                 # Deterministic LP Network RM solver (PuLP & SciPy HiGHS)
│   └── pricing_engine.py            # Simulation loop & event-driven recalculation trigger
├── main.py                          # Unified CLI entry point & comparative benchmarking
├── test_engine.py                   # OR Engine unit & integration test suite (8 tests)
├── test_api.py                      # FastAPI endpoint test suite (5 tests)
├── start_dashboard.bat              # 1-click Windows batch launcher (opens browser + server)
├── start_dashboard.ps1              # 1-click PowerShell launcher
├── requirements.txt                 # Dependencies: fastapi, uvicorn, numpy, pandas, scipy, pulp
└── README.md                        # Documentation & mathematical formulation
```

---

## 2. Mathematical Formulations

### A. Dynamic Programming: Finite-Horizon Markov Decision Process (MDP)

The single-resource dynamic pricing problem is modeled as a finite-horizon discrete-time MDP:

- **State Space**: $s = (c, t)$
  - $c \in \{0, 1, \dots, C\}$: Remaining unsold room capacity.
  - $t \in \{0, 1, \dots, T\}$: Decision epochs remaining until check-in.
- **Action Space**: $p \in \mathcal{P}$
  - Selection of a discrete price tier from a candidate set $\mathcal{P} = \{p_1, p_2, \dots, p_K\}$.
- **Probabilistic Demand & Transition**:
  - Potential booking requests arrive according to a non-homogeneous Poisson process. Over a discrete decision epoch $\Delta t$, the probability of a sale at price $p$ given competitor rate $p_{comp}$ and elasticity $\epsilon$ is:
    $$P(\text{booking} \mid p, p_{comp}) = 1 - \exp\left( -\frac{\lambda_0}{\text{daily\_steps}} \cdot \left(\frac{p}{p_{comp}}\right)^{-\epsilon} \right)$$
- **Bellman Optimality Equation**:
  $$V_t(c) = \max_{p \in \mathcal{P}} \Big\{ P(p, p_{comp}) \cdot \big[ p + V_{t-1}(c - 1) \big] + \big(1 - P(p, p_{comp})\big) \cdot V_{t-1}(c) \Big\}$$
  Rewriting in terms of marginal value:
  $$V_t(c) = V_{t-1}(c) + \max_{p \in \mathcal{P}} \Big\{ P(p, p_{comp}) \cdot \big[ p - \Delta V_{t-1}(c) \big] \Big\}$$
  where **$\Delta V_{t-1}(c) = V_{t-1}(c) - V_{t-1}(c - 1)$** represents the **Marginal Opportunity Cost (Capacity Bid Price)**.
- **Boundary Conditions**:
  $$V_0(c) = 0 \quad \forall c \ge 0 \quad \text{(Rooms expire at check-in)}$$
  $$V_t(0) = 0 \quad \forall t \ge 0 \quad \text{(Zero capacity remaining)}$$

### B. Linear Programming: Deterministic Network Revenue Management (NRM)

Formulates deterministic room allocation across customer segments $j \in \{\text{Budget}, \text{Standard}, \text{Premium}\}$ subject to hard competitor price index constraints:

$$\max_{x} \sum_{j \in \mathcal{S}} p_j \cdot x_j$$

**Subject to:**

1. **Physical Capacity Constraint:**
   $$\sum_{j \in \mathcal{S}} x_j \le c_{\text{remaining}} \quad \implies \quad \text{Dual Variable } \mu_{\text{capacity}} \text{ (Capacity Shadow Price / Bid Price)}$$
2. **Segment Demand Upper Bounds:**
   $$0 \le x_j \le D_j(p_j, p_{comp}) \quad \forall j \in \mathcal{S}$$
3. **Hard Competitor Price Index (CPI) Corridor:**
   $$\theta_{\min} \le \frac{\sum_j p_j x_j}{p_{comp} \sum_j x_j} \le \theta_{\max}$$
   Linearized as:
   $$\sum_{j \in \mathcal{S}} \big(p_j - \theta_{\min} \cdot p_{comp}\big) x_j \ge 0 \quad \text{(Brand floor constraint, e.g. } \theta_{\min} = 0.88\text{)}$$
   $$\sum_{j \in \mathcal{S}} \big(p_j - \theta_{\max} \cdot p_{comp}\big) x_j \le 0 \quad \text{(Competitive ceiling constraint, e.g. } \theta_{\max} = 1.25\text{)}$$

The solver backend supports both **PuLP** and **SciPy HiGHS**. The dual value of the capacity constraint $\mu_{\text{capacity}}$ serves as the threshold bid price for real-time acceptance.

---

## 3. Synthetic Dataset Generation (`market_conditions.csv`)

Generated 100% offline without external network dependencies:
- `days_to_checkin`: Horizon countdown from $T=30$ down to $1$.
- `current_occupancy`: Logistic S-curve modeling realistic market inventory build-up from 15% to ~85%.
- `competitor_avg_price`: Dynamic competitor rates that increase as check-in nears ($115 to $205).
- `price_elasticity_factor`: Compressing from 2.4 (early leisure bookers) down to 1.1 (last-minute business bookers).
- `base_demand_rate`: Poisson arrival rate peaking in the 2-week to 3-day lead window.

---

## 4. Trigger Mechanism (Event-Driven Recalculation)

The system simulates incoming inquiries over the 30-day horizon:
1. Every time a booking occurs, the state changes ($c \leftarrow c - 1$).
2. An event trigger immediately invokes `evaluate_realtime_pricing()`.
3. The engine computes:
   - Updated DP optimal price $p^*(c, t)$ and marginal capacity bid price $\Delta V_{t-1}(c)$.
   - Updated LP segment allocation $x_j^*$ and capacity shadow price $\mu_{\text{cap}}$.
4. Pricing adjustments dynamically surge when capacity tightens and soften when inventory is abundant.

---

## 5. Performance Benchmark Results (INR ₹)

Run on a standard 100-room property over 30 days:

| Metric | Baseline (Static Fixed) | Dynamic Prog (MDP) | Linear Prog (NRM) |
| :--- | :--- | :--- | :--- |
| **Total Realized Revenue** | **₹5,50,000.00** | **₹6,19,500.00** | **₹5,99,418.10** |
| **Rooms Sold / Occupancy** | 100 (100.0%) | 94 (94.0%) | 92 (92.0%) |
| **Average Daily Rate (ADR)**| ₹5,500.00 | ₹6,590.43 | ₹6,515.41 |
| **RevPAR** | ₹5,500.00 | ₹6,195.00 | ₹5,994.18 |
| **Price Range Offered** | ₹5,500.00 (Fixed) | ₹3,200 – ₹11,800 | ₹4,366 – ₹9,095 |
| **Projected Revenue Uplift (₹)** | ₹0.00 (Ref) | **+₹69,500.00** | **+₹49,418.10** |
| **Projected Revenue Uplift (%)** | **0.0%** | **+12.64%** | **+8.99%** |

### Key OR Insights:
1. **Dynamic Programming (+13.30% uplift)**: Captures consumer surplus by dynamically raising prices when lead time is short and remaining rooms are scarce (high bid price), while offering discounts early to secure baseline occupancy.
2. **Linear Programming (+5.03% uplift)**: Prevents inventory dilution by rationing capacity to high-yield segments (Premium/Corporate) while strictly honoring the competitor price index boundary.

---

## 6. How to Run

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Launch Web Dashboard & Local Backend (Recommended)
You can start the server with any of these methods:

**Method A: 1-Click Launch (Browser opens automatically)**
```powershell
.\start_dashboard.ps1
# Or double-click start_dashboard.bat
```

**Method B: Standard Python Execution**
```powershell
python server.py
# Or: python main.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.
Interactive API documentation is live at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

### Run Offline CLI Simulation Benchmark
```powershell
python main.py --cli
# Optional custom scenario:
python main.py --cli --rooms 120 --base-price 6000.0 --seed 99
```

### Run Unit and Integration Tests
```powershell
python test_engine.py   # Core OR solvers & demand model (8 tests)
python test_api.py      # FastAPI REST endpoints & schema validation (5 tests)
```

