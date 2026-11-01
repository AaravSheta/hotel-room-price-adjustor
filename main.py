"""
main.py
Real-Time Dynamic Hotel Room Pricing Engine & Local API Server.
Operations Research: Markov Decision Process (DP) & Network Revenue Management (LP)

Features:
  - Local FastAPI REST API Server with CORS (/optimize-price, /api/health, /api/market-feed)
  - Embedded Static Dashboard Serving at http://127.0.0.1:8000
  - CLI Offline Simulation & Benchmarking mode (--cli)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import argparse
import uvicorn

# Import FastAPI app from server module
from server import app, run_server
from src.data_generator import generate_market_conditions
from src.pricing_engine import HotelPricingEngine


def demonstrate_single_state_decision(engine: HotelPricingEngine, unsold_rooms: int = 25, days_left: int = 5):
    """
    Demonstrate granular OR decision-making at an exact snapshot in time.
    """
    print("\n" + "=" * 90)
    print(f" [SNAPSHOT EVALUATION] Real-Time State: {unsold_rooms} Unsold Rooms, T-{days_left} Days Remaining")
    print("=" * 90)

    row = engine.market_df[engine.market_df["days_to_checkin"] == days_left]
    if not row.empty:
        comp_price = float(row["competitor_avg_price"].iloc[0])
        elasticity = float(row["price_elasticity_factor"].iloc[0])
        base_demand = float(row["base_demand_rate"].iloc[0])
    else:
        comp_price, elasticity, base_demand = 6800.0, 1.45, 4.2

    eval_result = engine.evaluate_realtime_pricing(
        current_unsold_rooms=unsold_rooms,
        days_to_checkin=days_left,
        competitor_price=comp_price,
        base_demand_rate=base_demand,
        price_elasticity=elasticity
    )

    base = eval_result["baseline"]
    dp = eval_result["dynamic_programming"]
    lp = eval_result["linear_programming"]

    print(f" Market Conditions Feed:")
    print(f"   * Competitor Avg Rate:     ₹{comp_price:,.2f}")
    print(f"   * Current Occupancy:       {eval_result['occupancy_rate']*100:.1f}% ({engine.total_rooms - unsold_rooms}/{engine.total_rooms} rooms booked)")
    print(f"   * Price Elasticity Factor: {elasticity:.2f} (Inelastic last-minute market)")
    print(f"   * Base Demand Intensity:   {base_demand:.2f} arrivals/day")
    print("-" * 90)

    p_base_str = f"{base['booking_probability']*100:.1f}%"
    p_dp_str = f"{dp['booking_probability']*100:.1f}%"
    p_lp_str = f"{lp['booking_probability']*100:.1f}%"

    print(f"{'Decision Dimension':<30} | {'Baseline (Static)':<18} | {'Dynamic Prog (MDP)':<18} | {'Linear Prog (NRM)':<18}")
    print("-" * 90)
    print(f"{'Recommended Price (₹)':<30} | ₹{base['price']:<17,.2f} | ₹{dp['price']:<17,.2f} | ₹{lp['price']:<17,.2f}")
    print(f"{'Booking Probability / Epoch':<30} | {p_base_str:<18} | {p_dp_str:<18} | {p_lp_str:<18}")
    print(f"{'Capacity Bid Price (Shadow)':<30} | {'N/A (Static)':<18} | ₹{dp['marginal_bid_price']:<17,.2f} | ₹{lp['shadow_price_capacity']:<17,.2f}")
    print(f"{'Expected Future Value (₹)':<30} | {'N/A (Myopic)':<18} | ₹{dp['expected_future_revenue']:<17,.2f} | ₹{lp['weighted_avg_price'] * unsold_rooms:<17,.2f}")
    print(f"{'Competitor Price Index':<30} | {base['price']/comp_price:<18.3f} | {dp['price']/comp_price:<18.3f} | {lp['competitor_price_index']:<18.3f}")
    print("=" * 90)


def run_cli_mode(args):
    """Run full 30-day offline simulation in CLI."""
    if args.regen_data or not os.path.exists(args.dataset):
        print(f"[Phase 1] Generating synthetic market dataset (zero external APIs)...")
        generate_market_conditions(output_path=args.dataset, total_days=30, total_rooms=args.rooms, random_seed=args.seed)

    print(f"[Phase 2] Initializing OR Pricing Engine (DP Bellman MDP + LP Network RM)...")
    engine = HotelPricingEngine(
        total_rooms=args.rooms,
        baseline_fixed_price=args.base_price,
        dataset_path=args.dataset,
        random_seed=args.seed
    )

    demonstrate_single_state_decision(engine, unsold_rooms=20, days_left=4)

    print("\n[Phase 3] Launching Real-Time Booking Simulation Horizon (30 Days)...")
    engine.run_simulation(verbose=True, max_triggers_to_print=12)
    print("[Complete] Dynamic pricing engine execution finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Dynamic Hotel Room Pricing Engine & Local API Server")
    parser.add_argument("--cli", action="store_true", help="Run 30-day CLI simulation benchmark instead of web server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API Port (default: 8000)")
    parser.add_argument("--rooms", type=int, default=100, help="Total hotel room capacity (default: 100)")
    parser.add_argument("--base-price", type=float, default=5500.0, help="Baseline fixed room rate (₹) (default: 5500.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dataset", type=str, default="data/market_conditions.csv", help="Path to market CSV")
    parser.add_argument("--regen-data", action="store_true", help="Force regenerate synthetic market dataset")
    args = parser.parse_args()

    if args.cli:
        run_cli_mode(args)
    else:
        # Default behavior: Start local FastAPI backend server & serve dashboard
        run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
