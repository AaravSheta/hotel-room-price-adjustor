"""
pricing_engine.py
Real-Time Dynamic Hotel Room Pricing Engine & Simulation Loop.
Integrates Data Ingestion, Demand Modeling, Dynamic Programming (MDP),
Linear Programming (Network RM), and Event-Driven Price Recalculation Triggers.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.demand_model import HotelDemandModel
from src.dp_solver import DynamicProgrammingPricingSolver
from src.lp_solver import LinearProgrammingRevenueSolver
from src.data_generator import generate_market_conditions


class HotelPricingEngine:
    """
    Core Dynamic Pricing Engine for Hotel Revenue Management.

    Features:
      - Real-time data feed ingestion from synthetic market conditions.
      - Dynamic Programming (Bellman MDP backward induction).
      - Linear Programming (Deterministic Network RM with competitor constraints).
      - Simulated booking loop with event trigger recalculation upon room booking.
      - Head-to-head comparison of Baseline vs. DP vs. LP pricing strategies.
    """

    def __init__(
        self,
        total_rooms: int = 100,
        baseline_fixed_price: float = 140.0,
        dataset_path: str = "data/market_conditions.csv",
        random_seed: int = 42
    ):
        """
        Parameters:
            total_rooms: Total physical capacity of the hotel.
            baseline_fixed_price: Fixed price used by naive baseline strategy ($).
            dataset_path: Path to synthetic market conditions CSV.
            random_seed: Seed for stochastic customer booking generation.
        """
        self.total_rooms = total_rooms
        self.baseline_fixed_price = baseline_fixed_price
        self.dataset_path = dataset_path
        self.random_seed = random_seed

        # Initialize OR components
        self.demand_model = HotelDemandModel(time_step_hours=4.0)  # 6 decision epochs per day
        self.dp_solver = DynamicProgrammingPricingSolver(
            total_capacity=total_rooms,
            demand_model=self.demand_model
        )
        self.lp_solver = LinearProgrammingRevenueSolver(
            demand_model=self.demand_model,
            min_competitor_index=0.88,
            max_competitor_index=1.25
        )

        # Load or generate market feed
        self.market_df = self.load_or_generate_dataset()

    def load_or_generate_dataset(self) -> pd.DataFrame:
        """
        Load synthetic market dataset from disk; generate if absent.
        """
        if not os.path.exists(self.dataset_path):
            print(f"[Engine] Market dataset not found at '{self.dataset_path}'. Generating synthetic feed...")
            return generate_market_conditions(
                output_path=self.dataset_path,
                total_days=30,
                total_rooms=self.total_rooms,
                random_seed=self.random_seed
            )
        df = pd.read_csv(self.dataset_path)
        # Ensure sorted descending by days_to_checkin (30 down to 1)
        df = df.sort_values("days_to_checkin", ascending=False).reset_index(drop=True)
        return df

    def evaluate_realtime_pricing(
        self,
        current_unsold_rooms: int,
        days_to_checkin: int,
        competitor_price: float,
        base_demand_rate: float,
        price_elasticity: float
    ) -> Dict[str, Any]:
        """
        Evaluate real-time pricing recommendations from both OR models.

        Triggered dynamically whenever a room is booked or market conditions shift.
        """
        # 1. Baseline fixed price
        p_base = self.baseline_fixed_price
        prob_base = self.demand_model.booking_probability_poisson(
            price=p_base,
            competitor_price=competitor_price,
            base_daily_demand=base_demand_rate,
            elasticity=price_elasticity
        )

        # 2. Dynamic Programming (Bellman MDP)
        dp_result = self.dp_solver.get_optimal_price(
            current_unsold_rooms=current_unsold_rooms,
            days_to_checkin=days_to_checkin,
            competitor_price=competitor_price,
            base_daily_demand=base_demand_rate,
            elasticity=price_elasticity
        )

        # 3. Linear Programming (Deterministic Network RM)
        lp_result = self.lp_solver.solve(
            remaining_capacity=current_unsold_rooms,
            days_to_checkin=days_to_checkin,
            competitor_price=competitor_price,
            base_daily_demand=base_demand_rate,
            elasticity=price_elasticity,
            backend="pulp"
        )
        prob_lp = self.demand_model.booking_probability_poisson(
            price=lp_result["recommended_dynamic_price"],
            competitor_price=competitor_price,
            base_daily_demand=base_demand_rate,
            elasticity=price_elasticity
        ) if current_unsold_rooms > 0 else 0.0

        return {
            "days_to_checkin": days_to_checkin,
            "unsold_rooms": current_unsold_rooms,
            "occupancy_rate": round(1.0 - (current_unsold_rooms / self.total_rooms), 3),
            "competitor_price": competitor_price,
            "elasticity": price_elasticity,
            "base_demand": base_demand_rate,
            "baseline": {
                "price": p_base,
                "booking_probability": round(prob_base, 4)
            },
            "dynamic_programming": {
                "price": dp_result["optimal_price"],
                "marginal_bid_price": dp_result["marginal_bid_price"],
                "expected_future_revenue": dp_result["expected_future_revenue"],
                "booking_probability": dp_result["booking_probability"]
            },
            "linear_programming": {
                "price": lp_result["recommended_dynamic_price"],
                "shadow_price_capacity": lp_result["shadow_price_capacity"],
                "weighted_avg_price": lp_result["weighted_avg_price"],
                "competitor_price_index": lp_result["competitor_price_index"],
                "allocation": lp_result["allocation"],
                "booking_probability": round(prob_lp, 4)
            }
        }

    def run_simulation(
        self,
        verbose: bool = True,
        max_triggers_to_print: int = 15
    ) -> Dict[str, Any]:
        """
        Execute simulated real-time booking loop over the booking horizon.

        Simulates incoming customer booking requests across daily epochs.
        Tracks baseline, DP, and LP strategies under the identical customer arrival stream.
        Recalculates optimal price every time a room gets booked (trigger mechanism).
        """
        np.random.seed(self.random_seed)

        # Initial state for each strategy: all rooms initially available
        # We start with total capacity C
        strategies = {
            "Baseline": {"unsold": self.total_rooms, "revenue": 0.0, "bookings": 0, "prices": []},
            "DP": {"unsold": self.total_rooms, "revenue": 0.0, "bookings": 0, "prices": []},
            "LP": {"unsold": self.total_rooms, "revenue": 0.0, "bookings": 0, "prices": []}
        }

        trigger_log: List[Dict[str, Any]] = []
        total_recalculations = 0

        if verbose:
            print("=" * 90)
            print(" HOTEL ROOM PRICING ENGINE: REAL-TIME SIMULATION LOOP START")
            print(f" Total Capacity: {self.total_rooms} rooms | Horizon: 30 days | Decision steps/day: 6")
            print(f" Fixed Baseline Price: ${self.baseline_fixed_price:.2f}")
            print("=" * 90)

        # Loop through each day from 30 down to 1
        for _, row in self.market_df.iterrows():
            day = int(row["days_to_checkin"])
            p_comp = float(row["competitor_avg_price"])
            elasticity = float(row["price_elasticity_factor"])
            base_demand = float(row["base_demand_rate"])
            epochs = int(self.demand_model.daily_steps)  # 6 decision epochs per day

            for epoch in range(epochs):
                # Random shock representing customer arrival trial
                rand_draw = np.random.uniform(0.0, 1.0)

                # -------------------------------------------------------------
                # 1. BASELINE STRATEGY
                # -------------------------------------------------------------
                if strategies["Baseline"]["unsold"] > 0:
                    p_base = self.baseline_fixed_price
                    prob_base = self.demand_model.booking_probability_poisson(
                        price=p_base,
                        competitor_price=p_comp,
                        base_daily_demand=base_demand,
                        elasticity=elasticity
                    )
                    if rand_draw < prob_base:
                        strategies["Baseline"]["unsold"] -= 1
                        strategies["Baseline"]["revenue"] += p_base
                        strategies["Baseline"]["bookings"] += 1
                        strategies["Baseline"]["prices"].append(p_base)

                # -------------------------------------------------------------
                # 2. DYNAMIC PROGRAMMING (MDP) STRATEGY
                # -------------------------------------------------------------
                if strategies["DP"]["unsold"] > 0:
                    # Query current optimal DP price
                    dp_eval = self.dp_solver.get_optimal_price(
                        current_unsold_rooms=strategies["DP"]["unsold"],
                        days_to_checkin=day,
                        competitor_price=p_comp,
                        base_daily_demand=base_demand,
                        elasticity=elasticity
                    )
                    p_dp = dp_eval["optimal_price"]
                    prob_dp = dp_eval["booking_probability"]

                    if rand_draw < prob_dp:
                        prev_unsold = strategies["DP"]["unsold"]
                        strategies["DP"]["unsold"] -= 1
                        strategies["DP"]["revenue"] += p_dp
                        strategies["DP"]["bookings"] += 1
                        strategies["DP"]["prices"].append(p_dp)
                        total_recalculations += 1

                        # TRIGGER MECHANISM: Room booked -> Occupancy shifted -> Trigger price recalculation
                        new_eval = self.dp_solver.get_optimal_price(
                            current_unsold_rooms=strategies["DP"]["unsold"],
                            days_to_checkin=day,
                            competitor_price=p_comp,
                            base_daily_demand=base_demand,
                            elasticity=elasticity
                        )
                        new_p_dp = new_eval["optimal_price"]
                        bid_p = new_eval["marginal_bid_price"]

                        occ_pct = round((self.total_rooms - strategies["DP"]["unsold"]) / self.total_rooms * 100, 1)

                        event_entry = {
                            "day": day,
                            "epoch": epoch + 1,
                            "event": "BOOKING_CONFIRMED",
                            "strategy": "DP",
                            "price_sold": p_dp,
                            "new_optimal_price": new_p_dp,
                            "bid_price": bid_p,
                            "competitor_price": p_comp,
                            "remaining_rooms": strategies["DP"]["unsold"],
                            "occupancy_pct": occ_pct
                        }
                        trigger_log.append(event_entry)

                        if verbose and len(trigger_log) <= max_triggers_to_print:
                            print(
                                f"  [TRIGGER T-{day:02d} Ep {epoch+1}] Booked at ${p_dp:>6.2f} | "
                                f"Occ: {occ_pct:>5.1f}% | New DP Price: ${new_p_dp:>6.2f} "
                                f"(Bid Price: ${bid_p:>5.2f}, Comp: ${p_comp:>6.2f})"
                            )

                # -------------------------------------------------------------
                # 3. LINEAR PROGRAMMING (NETWORK RM) STRATEGY
                # -------------------------------------------------------------
                if strategies["LP"]["unsold"] > 0:
                    lp_eval = self.lp_solver.solve(
                        remaining_capacity=strategies["LP"]["unsold"],
                        days_to_checkin=day,
                        competitor_price=p_comp,
                        base_daily_demand=base_demand,
                        elasticity=elasticity,
                        backend="pulp"
                    )
                    p_lp = lp_eval["recommended_dynamic_price"]
                    prob_lp = self.demand_model.booking_probability_poisson(
                        price=p_lp,
                        competitor_price=p_comp,
                        base_daily_demand=base_demand,
                        elasticity=elasticity
                    )
                    if rand_draw < prob_lp:
                        strategies["LP"]["unsold"] -= 1
                        strategies["LP"]["revenue"] += p_lp
                        strategies["LP"]["bookings"] += 1
                        strategies["LP"]["prices"].append(p_lp)

        # Compile final performance report
        summary_results = self._compile_summary(strategies, total_recalculations, trigger_log)

        if verbose:
            self._print_executive_summary(summary_results)

        return summary_results

    def _compile_summary(
        self,
        strategies: Dict[str, Dict[str, Any]],
        total_recalculations: int,
        trigger_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compile comparative performance metrics and projected revenue increases.
        """
        base_rev = strategies["Baseline"]["revenue"]
        dp_rev = strategies["DP"]["revenue"]
        lp_rev = strategies["LP"]["revenue"]

        base_bookings = strategies["Baseline"]["bookings"]
        dp_bookings = strategies["DP"]["bookings"]
        lp_bookings = strategies["LP"]["bookings"]

        base_adr = base_rev / base_bookings if base_bookings > 0 else 0.0
        dp_adr = dp_rev / dp_bookings if dp_bookings > 0 else 0.0
        lp_adr = lp_rev / lp_bookings if lp_bookings > 0 else 0.0

        base_revpar = base_rev / self.total_rooms
        dp_revpar = dp_rev / self.total_rooms
        lp_revpar = lp_rev / self.total_rooms

        dp_rev_increase_abs = dp_rev - base_rev
        dp_rev_increase_pct = (dp_rev_increase_abs / base_rev * 100.0) if base_rev > 0 else 0.0

        lp_rev_increase_abs = lp_rev - base_rev
        lp_rev_increase_pct = (lp_rev_increase_abs / base_rev * 100.0) if base_rev > 0 else 0.0

        return {
            "total_rooms": self.total_rooms,
            "baseline": {
                "total_revenue": round(base_rev, 2),
                "rooms_sold": base_bookings,
                "occupancy_rate": round(base_bookings / self.total_rooms, 3),
                "adr": round(base_adr, 2),
                "revpar": round(base_revpar, 2),
                "avg_price": round(float(np.mean(strategies["Baseline"]["prices"] or [self.baseline_fixed_price])), 2)
            },
            "dynamic_programming": {
                "total_revenue": round(dp_rev, 2),
                "rooms_sold": dp_bookings,
                "occupancy_rate": round(dp_bookings / self.total_rooms, 3),
                "adr": round(dp_adr, 2),
                "revpar": round(dp_revpar, 2),
                "min_price": round(float(np.min(strategies["DP"]["prices"] or [0.0])), 2),
                "max_price": round(float(np.max(strategies["DP"]["prices"] or [0.0])), 2),
                "avg_price": round(float(np.mean(strategies["DP"]["prices"] or [0.0])), 2),
                "revenue_increase_abs": round(dp_rev_increase_abs, 2),
                "revenue_increase_pct": round(dp_rev_increase_pct, 2)
            },
            "linear_programming": {
                "total_revenue": round(lp_rev, 2),
                "rooms_sold": lp_bookings,
                "occupancy_rate": round(lp_bookings / self.total_rooms, 3),
                "adr": round(lp_adr, 2),
                "revpar": round(lp_revpar, 2),
                "min_price": round(float(np.min(strategies["LP"]["prices"] or [0.0])), 2),
                "max_price": round(float(np.max(strategies["LP"]["prices"] or [0.0])), 2),
                "avg_price": round(float(np.mean(strategies["LP"]["prices"] or [0.0])), 2),
                "revenue_increase_abs": round(lp_rev_increase_abs, 2),
                "revenue_increase_pct": round(lp_rev_increase_pct, 2)
            },
            "simulation_stats": {
                "total_trigger_events": total_recalculations,
                "trigger_log_sample_count": len(trigger_log)
            }
        }

    def _print_executive_summary(self, summary: Dict[str, Any]) -> None:
        """
        Print clean comparative results table.
        """
        b = summary["baseline"]
        dp = summary["dynamic_programming"]
        lp = summary["linear_programming"]

        print("\n" + "=" * 90)
        print("          REAL-TIME HOTEL PRICING ENGINE: FINAL PERFORMANCE BENCHMARK")
        print("=" * 90)
        header = f"{'Metric':<28} | {'Baseline (Static)':<18} | {'Dynamic Prog (MDP)':<18} | {'Linear Prog (NRM)':<18}"
        print(header)
        print("-" * 90)

        rows = [
            ("Total Realized Revenue ($)", f"${b['total_revenue']:,.2f}", f"${dp['total_revenue']:,.2f}", f"${lp['total_revenue']:,.2f}"),
            ("Rooms Sold / Occupancy", f"{b['rooms_sold']} ({b['occupancy_rate']*100:.1f}%)", f"{dp['rooms_sold']} ({dp['occupancy_rate']*100:.1f}%)", f"{lp['rooms_sold']} ({lp['occupancy_rate']*100:.1f}%)"),
            ("Average Daily Rate (ADR)", f"${b['adr']:.2f}", f"${dp['adr']:.2f}", f"${lp['adr']:.2f}"),
            ("RevPAR (Rev / Total Cap)", f"${b['revpar']:.2f}", f"${dp['revpar']:.2f}", f"${lp['revpar']:.2f}"),
            ("Price Range Seen ($)", f"${b['avg_price']:.2f}", f"${dp['min_price']:.2f} - ${dp['max_price']:.2f}", f"${lp['min_price']:.2f} - ${lp['max_price']:.2f}"),
            ("Revenue Uplift ($)", "$0.00 (Ref)", f"+${dp['revenue_increase_abs']:,.2f}", f"+${lp['revenue_increase_abs']:,.2f}"),
            ("Revenue Uplift (%)", "0.0%", f"+{dp['revenue_increase_pct']:.2f}%", f"+{lp['revenue_increase_pct']:.2f}%")
        ]

        for label, v_b, v_dp, v_lp in rows:
            print(f"{label:<28} | {v_b:<18} | {v_dp:<18} | {v_lp:<18}")

        print("=" * 90)
        print(
            f" [INSIGHT] DP achieved +{dp['revenue_increase_pct']:.2f}% revenue uplift by capturing consumer surplus\n"
            f"           via Bellman opportunity cost bid pricing as capacity tightened.\n"
            f" [INSIGHT] LP achieved +{lp['revenue_increase_pct']:.2f}% revenue uplift via customer segment allocation\n"
            f"           under strict competitor price index constraints."
        )
        print("=" * 90 + "\n")

