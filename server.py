"""
server.py
Local FastAPI Backend for Hotel Room Dynamic Pricing Engine.
Routes incoming market parameter JSON payloads through DP (Bellman MDP)
and LP (Deterministic Network RM) solvers with CORS enabled.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from src.demand_model import HotelDemandModel
from src.dp_solver import DynamicProgrammingPricingSolver
from src.lp_solver import LinearProgrammingRevenueSolver
from src.data_generator import generate_market_conditions

# Initialize FastAPI application
app = FastAPI(
    title="Hotel Room Dynamic Pricing Engine API",
    description="Operations Research Dynamic Pricing Local REST API",
    version="1.0.0"
)

# Configure CORS for local frontend execution
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows localhost, file://, VS Code Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure synthetic market conditions exist
market_csv_path = "data/market_conditions.csv"
if not os.path.exists(market_csv_path):
    generate_market_conditions(output_path=market_csv_path, total_days=30, total_rooms=100)

market_df = pd.read_csv(market_csv_path)

# Initialize OR Solvers
demand_model = HotelDemandModel(time_step_hours=4.0)
dp_solver = DynamicProgrammingPricingSolver(total_capacity=100, demand_model=demand_model)
lp_solver = LinearProgrammingRevenueSolver(demand_model=demand_model)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class PricingRequest(BaseModel):
    days_to_checkin: int = Field(..., ge=1, le=100, description="Lead time in days remaining until check-in")
    current_occupancy: float = Field(..., ge=0.0, le=100.0, description="Occupancy rate (either 0.0-1.0 fraction or 0-100 percentage)")
    competitor_avg_price: float = Field(..., gt=0.0, description="Current competitor average benchmark rate (₹)")
    base_price: Optional[float] = Field(default=5500.0, gt=0.0, description="Initial baseline fixed price benchmark (₹)")
    total_rooms: Optional[int] = Field(default=100, ge=1, description="Total room inventory capacity")
    price_elasticity_factor: Optional[float] = Field(default=None, gt=0.0, description="Price elasticity factor (optional)")
    base_demand_rate: Optional[float] = Field(default=None, gt=0.0, description="Base Poisson arrival intensity per day (optional)")


class PricingResponse(BaseModel):
    status: str
    inputs: Dict[str, Any]
    baseline: Dict[str, Any]
    dynamic_programming: Dict[str, Any]
    linear_programming: Dict[str, Any]
    simulation_summary: Dict[str, Any]


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend connectivity."""
    return {
        "status": "healthy",
        "service": "Hotel Dynamic Pricing Engine API",
        "solvers": ["Bellman MDP Dynamic Programming", "PuLP & SciPy Network LP"]
    }


@app.get("/api/market-feed")
def get_market_feed():
    """Returns the 30-day synthetic market dataset for historical visualization."""
    return market_df.to_dict(orient="records")


@app.post("/optimize-price", response_model=PricingResponse)
def optimize_price(payload: PricingRequest):
    """
    Core Optimization Endpoint.
    Accepts current market conditions and routes them through both DP and LP solvers.
    """
    try:
        # Normalize occupancy: if passed as > 1.0 (e.g. 75), convert to fraction 0.75
        occ_frac = payload.current_occupancy / 100.0 if payload.current_occupancy > 1.0 else payload.current_occupancy
        occ_frac = max(0.0, min(1.0, occ_frac))

        total_rooms = payload.total_rooms or 100
        rooms_sold = int(round(occ_frac * total_rooms))
        unsold_rooms = max(0, total_rooms - rooms_sold)

        days_left = payload.days_to_checkin
        p_comp = payload.competitor_avg_price
        p_base = payload.base_price or 5500.0

        # Estimate elasticity & demand from synthetic market curve if not explicitly provided
        matched_row = market_df[market_df["days_to_checkin"] == days_left]
        if not matched_row.empty:
            def_elasticity = float(matched_row["price_elasticity_factor"].iloc[0])
            def_demand = float(matched_row["base_demand_rate"].iloc[0])
        else:
            # Formula-based interpolation
            norm_t = (30 - min(30, days_left)) / 30.0
            def_elasticity = round(2.4 - 1.25 * (norm_t ** 0.8), 2)
            def_demand = round(2.0 + 2.5 * np.sin(np.pi * norm_t) + 0.5 * norm_t, 2)

        elasticity = payload.price_elasticity_factor if payload.price_elasticity_factor is not None else def_elasticity
        base_demand = payload.base_demand_rate if payload.base_demand_rate is not None else def_demand

        # ---------------------------------------------------------------------
        # 1. BASELINE PRICING CALCULATION
        # ---------------------------------------------------------------------
        prob_base = demand_model.booking_probability_poisson(
            price=p_base,
            competitor_price=p_comp,
            base_daily_demand=base_demand,
            elasticity=elasticity
        )
        # Expected bookings for baseline over remaining decision epochs (capped at unsold)
        epochs_left = max(1, int(days_left * demand_model.daily_steps))
        expected_base_bookings = min(float(unsold_rooms), prob_base * epochs_left)
        base_projected_revenue = round(expected_base_bookings * p_base, 2)

        # ---------------------------------------------------------------------
        # 2. DYNAMIC PROGRAMMING (BELLMAN MDP) SOLVER
        # ---------------------------------------------------------------------
        dp_result = dp_solver.get_optimal_price(
            current_unsold_rooms=unsold_rooms,
            days_to_checkin=days_left,
            competitor_price=p_comp,
            base_daily_demand=base_demand,
            elasticity=elasticity
        )
        p_dp = dp_result["optimal_price"]
        prob_dp = dp_result["booking_probability"]
        bid_price_dp = dp_result["marginal_bid_price"]
        dp_expected_rev = dp_result["expected_future_revenue"]

        # Revenue lift vs baseline
        dp_lift_abs = round(dp_expected_rev - base_projected_revenue, 2)
        dp_lift_pct = round((dp_lift_abs / base_projected_revenue * 100.0) if base_projected_revenue > 0 else 0.0, 2)

        dp_strategy_note = (
            f"Opportunity cost (bid price) is ₹{bid_price_dp:,.2f}. "
            f"{'Capacity is scarce; rate raised to capture consumer surplus.' if p_dp > p_comp else 'Rate positioned competitively to accelerate booking velocity.'}"
        )

        # ---------------------------------------------------------------------
        # 3. LINEAR PROGRAMMING (NETWORK REVENUE MANAGEMENT) SOLVER
        # ---------------------------------------------------------------------
        lp_result = lp_solver.solve(
            remaining_capacity=unsold_rooms,
            days_to_checkin=days_left,
            competitor_price=p_comp,
            base_daily_demand=base_demand,
            elasticity=elasticity,
            backend="pulp"
        )
        p_lp = lp_result["recommended_dynamic_price"]
        lp_shadow = lp_result["shadow_price_capacity"]
        lp_rev = lp_result["total_revenue"]
        prob_lp = demand_model.booking_probability_poisson(
            price=p_lp,
            competitor_price=p_comp,
            base_daily_demand=base_demand,
            elasticity=elasticity
        ) if unsold_rooms > 0 else 0.0

        lp_lift_abs = round(lp_rev - base_projected_revenue, 2)
        lp_lift_pct = round((lp_lift_abs / base_projected_revenue * 100.0) if base_projected_revenue > 0 else 0.0, 2)

        lp_strategy_note = (
            f"Allocated {lp_result['total_rooms_allocated']} rooms across segments with CPI {lp_result['competitor_price_index']:.3f} "
            f"under strict brand floor and ceiling constraints."
        )

        return PricingResponse(
            status="success",
            inputs={
                "days_to_checkin": days_left,
                "current_occupancy_pct": round(occ_frac * 100.0, 1),
                "unsold_rooms": unsold_rooms,
                "rooms_sold": rooms_sold,
                "total_rooms": total_rooms,
                "competitor_avg_price": round(p_comp, 2),
                "baseline_price": round(p_base, 2),
                "price_elasticity_factor": round(elasticity, 2),
                "base_demand_rate": round(base_demand, 2)
            },
            baseline={
                "price": round(p_base, 2),
                "booking_probability": round(prob_base, 4),
                "expected_bookings": round(expected_base_bookings, 1),
                "projected_revenue": base_projected_revenue
            },
            dynamic_programming={
                "price": round(p_dp, 2),
                "marginal_bid_price": round(bid_price_dp, 2),
                "booking_probability": round(prob_dp, 4),
                "expected_future_revenue": round(dp_expected_rev, 2),
                "revenue_lift_abs": dp_lift_abs,
                "revenue_lift_pct": dp_lift_pct,
                "strategy_note": dp_strategy_note
            },
            linear_programming={
                "price": round(p_lp, 2),
                "shadow_price_capacity": round(lp_shadow, 2),
                "weighted_avg_price": lp_result["weighted_avg_price"],
                "competitor_price_index": lp_result["competitor_price_index"],
                "allocation": lp_result["allocation"],
                "booking_probability": round(prob_lp, 4),
                "projected_revenue": lp_rev,
                "revenue_lift_abs": lp_lift_abs,
                "revenue_lift_pct": lp_lift_pct,
                "strategy_note": lp_strategy_note,
                "segment_prices": lp_result.get("segment_prices", {})
            },
            simulation_summary={
                "dp_projected_uplift_pct": dp_lift_pct,
                "lp_projected_uplift_pct": lp_lift_pct,
                "max_recommended_price": round(max(p_dp, p_lp), 2)
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


from fastapi.responses import HTMLResponse, Response

# -----------------------------------------------------------------------------
# Static Files & Dashboard Mounting
# -----------------------------------------------------------------------------
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")

@app.get("/", response_class=HTMLResponse)
def serve_frontend_root():
    """Serve main frontend dashboard directly at root URL."""
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Frontend index.html not found</h3>", status_code=404)

@app.get("/app.js")
def serve_app_js():
    js_file = os.path.join(frontend_dir, "app.js")
    if os.path.exists(js_file):
        with open(js_file, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="application/javascript")
    return Response(content="", status_code=404)

@app.get("/style.css")
def serve_style_css():
    css_file = os.path.join(frontend_dir, "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/css")
    return Response(content="", status_code=404)


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Start local uvicorn server."""
    print("=" * 70)
    print(" HOTEL ROOM DYNAMIC PRICING ENGINE - LOCAL FASTAPI SERVER")
    print(f" Web Dashboard: http://{host}:{port}")
    print(f" REST API Docs:  http://{host}:{port}/docs")
    print("=" * 70)
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
