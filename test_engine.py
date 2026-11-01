"""
test_engine.py
Comprehensive Unit & Integration Test Suite for Dynamic Hotel Room Pricing Engine (INR ₹).
Tests Data Ingestion, Demand Modeling, DP (Bellman MDP), LP (Network RM), and Simulation Triggers.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import unittest
import numpy as np
import pandas as pd

from src.data_generator import generate_market_conditions
from src.demand_model import HotelDemandModel
from src.dp_solver import DynamicProgrammingPricingSolver
from src.lp_solver import LinearProgrammingRevenueSolver
from src.pricing_engine import HotelPricingEngine


class TestHotelPricingEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_csv_path = "data/test_market_conditions.csv"
        cls.df = generate_market_conditions(
            output_path=cls.test_csv_path,
            total_days=30,
            total_rooms=100,
            random_seed=42
        )
        cls.demand_model = HotelDemandModel(time_step_hours=4.0)
        cls.dp_solver = DynamicProgrammingPricingSolver(total_capacity=100, demand_model=cls.demand_model)
        cls.lp_solver = LinearProgrammingRevenueSolver(demand_model=cls.demand_model)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv_path):
            os.remove(cls.test_csv_path)

    def test_synthetic_dataset_structure(self):
        self.assertEqual(len(self.df), 30)
        required_cols = [
            "days_to_checkin",
            "current_occupancy",
            "competitor_avg_price",
            "price_elasticity_factor",
            "base_demand_rate"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns)

        self.assertFalse(self.df.isnull().values.any(), "Dataset must have no null values.")
        self.assertTrue((self.df["current_occupancy"] >= 0.0).all() and (self.df["current_occupancy"] <= 1.0).all())
        self.assertTrue((self.df["competitor_avg_price"] > 1000.0).all())
        self.assertTrue((self.df["price_elasticity_factor"] > 0.0).all())
        self.assertTrue((self.df["base_demand_rate"] > 0.0).all())

    def test_demand_model_monotonicity_and_bounds(self):
        p_low = 4500.0
        p_high = 8500.0
        comp_price = 6500.0
        base_demand = 3.5
        elasticity = 1.6

        prob_low = self.demand_model.booking_probability_poisson(p_low, comp_price, base_demand, elasticity)
        prob_high = self.demand_model.booking_probability_poisson(p_high, comp_price, base_demand, elasticity)

        self.assertGreater(prob_low, prob_high, "Booking probability must decrease with higher prices.")
        self.assertGreaterEqual(prob_low, 0.0)
        self.assertLessEqual(prob_low, 1.0)
        self.assertGreaterEqual(prob_high, 0.0)
        self.assertLessEqual(prob_high, 1.0)

    def test_demand_elasticity_sensitivity(self):
        p_above_comp = 8000.0
        comp_price = 6500.0
        base_demand = 3.5

        prob_inelastic = self.demand_model.booking_probability_poisson(p_above_comp, comp_price, base_demand, elasticity=1.1)
        prob_elastic = self.demand_model.booking_probability_poisson(p_above_comp, comp_price, base_demand, elasticity=2.5)

        self.assertGreater(prob_inelastic, prob_elastic)

    def test_dp_boundary_conditions(self):
        res_zero_rooms = self.dp_solver.get_optimal_price(
            current_unsold_rooms=0,
            days_to_checkin=10,
            competitor_price=6500.0,
            base_daily_demand=3.0,
            elasticity=1.5
        )
        self.assertEqual(res_zero_rooms["optimal_price"], 0.0)
        self.assertEqual(res_zero_rooms["expected_future_revenue"], 0.0)
        self.assertEqual(res_zero_rooms["marginal_bid_price"], 0.0)

    def test_dp_scarcity_reaction(self):
        res_abundant = self.dp_solver.get_optimal_price(
            current_unsold_rooms=60,
            days_to_checkin=5,
            competitor_price=6800.0,
            base_daily_demand=3.0,
            elasticity=1.5
        )
        res_scarce = self.dp_solver.get_optimal_price(
            current_unsold_rooms=4,
            days_to_checkin=5,
            competitor_price=6800.0,
            base_daily_demand=3.0,
            elasticity=1.5
        )

        self.assertGreaterEqual(res_scarce["optimal_price"], res_abundant["optimal_price"])
        self.assertGreaterEqual(res_scarce["marginal_bid_price"], res_abundant["marginal_bid_price"])

    def test_lp_capacity_and_competitor_index_constraints(self):
        c_rem = 25
        p_comp = 6800.0
        res_pulp = self.lp_solver.solve_pulp(
            remaining_capacity=c_rem,
            days_to_checkin=8,
            competitor_price=p_comp,
            base_daily_demand=3.8,
            elasticity=1.7
        )

        self.assertLessEqual(res_pulp["total_rooms_allocated"], c_rem + 1e-4)

        if res_pulp["total_rooms_allocated"] > 0:
            self.assertGreaterEqual(res_pulp["competitor_price_index"], self.lp_solver.min_cpi - 1e-3)
            self.assertLessEqual(res_pulp["competitor_price_index"], self.lp_solver.max_cpi + 1e-3)

        self.assertGreaterEqual(res_pulp["shadow_price_capacity"], 0.0)

    def test_lp_pulp_and_scipy_consistency(self):
        res_pulp = self.lp_solver.solve(
            remaining_capacity=30,
            days_to_checkin=12,
            competitor_price=6200.0,
            base_daily_demand=4.0,
            elasticity=1.8,
            backend="pulp"
        )
        res_scipy = self.lp_solver.solve(
            remaining_capacity=30,
            days_to_checkin=12,
            competitor_price=6200.0,
            base_daily_demand=4.0,
            elasticity=1.8,
            backend="scipy"
        )

        self.assertAlmostEqual(res_pulp["total_revenue"], res_scipy["total_revenue"], delta=10.0)
        self.assertAlmostEqual(res_pulp["total_rooms_allocated"], res_scipy["total_rooms_allocated"], delta=0.5)

    def test_full_simulation_pipeline(self):
        engine = HotelPricingEngine(
            total_rooms=50,
            baseline_fixed_price=5500.0,
            dataset_path=self.test_csv_path,
            random_seed=123
        )
        results = engine.run_simulation(verbose=False)

        self.assertIn("baseline", results)
        self.assertIn("dynamic_programming", results)
        self.assertIn("linear_programming", results)

        b_rev = results["baseline"]["total_revenue"]
        dp_rev = results["dynamic_programming"]["total_revenue"]
        lp_rev = results["linear_programming"]["total_revenue"]

        self.assertGreater(b_rev, 0.0)
        self.assertGreater(dp_rev, 0.0)
        self.assertGreater(lp_rev, 0.0)
        self.assertGreater(results["simulation_stats"]["total_trigger_events"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
