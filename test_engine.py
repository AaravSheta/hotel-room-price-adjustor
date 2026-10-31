"""
test_engine.py
Comprehensive Unit & Integration Test Suite for Dynamic Hotel Room Pricing Engine.
Tests Data Ingestion, Demand Modeling, DP (Bellman MDP), LP (Network RM), and Simulation Triggers.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import unittest
import numpy as np
import pandas as pd

from src.data_generator import generate_market_conditions
from src.demand_model import HotelDemandModel
from src.dp_solver import DynamicProgrammingPricingSolver
from src.lp_solver import LinearProgrammingRevenueSolver
from src.pricing_engine import HotelPricingEngine


class TestHotelPricingEngine(unittest.TestCase):
    """
    Test suite covering core Operations Research components and engineering specifications.
    """

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

    # -------------------------------------------------------------------------
    # 1. Dataset Generation & Schema Tests
    # -------------------------------------------------------------------------
    def test_synthetic_dataset_structure(self):
        """Verify generated dataset contains exact required columns, rows, and valid ranges."""
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
        self.assertTrue((self.df["competitor_avg_price"] > 50.0).all())
        self.assertTrue((self.df["price_elasticity_factor"] > 0.0).all())
        self.assertTrue((self.df["base_demand_rate"] > 0.0).all())

    # -------------------------------------------------------------------------
    # 2. Demand Model Tests
    # -------------------------------------------------------------------------
    def test_demand_model_monotonicity_and_bounds(self):
        """Higher prices must result in lower booking probability (Law of Demand)."""
        p_low = 100.0
        p_high = 200.0
        comp_price = 150.0
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
        """Higher price elasticity must cause steeper drop in demand for price increases."""
        p_above_comp = 180.0
        comp_price = 150.0
        base_demand = 3.5

        prob_inelastic = self.demand_model.booking_probability_poisson(p_above_comp, comp_price, base_demand, elasticity=1.1)
        prob_elastic = self.demand_model.booking_probability_poisson(p_above_comp, comp_price, base_demand, elasticity=2.5)

        self.assertGreater(prob_inelastic, prob_elastic, "Inelastic customers should book at higher rates than elastic customers.")

    # -------------------------------------------------------------------------
    # 3. Dynamic Programming (Bellman MDP) Tests
    # -------------------------------------------------------------------------
    def test_dp_boundary_conditions(self):
        """Bellman equation boundary condition: V(c=0, t) == 0 for all t."""
        res_zero_rooms = self.dp_solver.get_optimal_price(
            current_unsold_rooms=0,
            days_to_checkin=10,
            competitor_price=150.0,
            base_daily_demand=3.0,
            elasticity=1.5
        )
        self.assertEqual(res_zero_rooms["optimal_price"], 0.0)
        self.assertEqual(res_zero_rooms["expected_future_revenue"], 0.0)
        self.assertEqual(res_zero_rooms["marginal_bid_price"], 0.0)

    def test_dp_scarcity_reaction(self):
        """When capacity is scarce, DP marginal bid price and price should surge."""
        res_abundant = self.dp_solver.get_optimal_price(
            current_unsold_rooms=60,
            days_to_checkin=5,
            competitor_price=160.0,
            base_daily_demand=3.0,
            elasticity=1.5
        )
        res_scarce = self.dp_solver.get_optimal_price(
            current_unsold_rooms=4,
            days_to_checkin=5,
            competitor_price=160.0,
            base_daily_demand=3.0,
            elasticity=1.5
        )

        self.assertGreaterEqual(
            res_scarce["optimal_price"],
            res_abundant["optimal_price"],
            "Scarce inventory should yield higher or equal optimal price."
        )
        self.assertGreaterEqual(
            res_scarce["marginal_bid_price"],
            res_abundant["marginal_bid_price"],
            "Scarce inventory should carry higher capacity opportunity cost (bid price)."
        )

    # -------------------------------------------------------------------------
    # 4. Linear Programming (Network RM) Tests
    # -------------------------------------------------------------------------
    def test_lp_capacity_and_competitor_index_constraints(self):
        """LP allocation must not exceed capacity and must satisfy hard competitor price index corridor."""
        c_rem = 25
        p_comp = 160.0
        res_pulp = self.lp_solver.solve_pulp(
            remaining_capacity=c_rem,
            days_to_checkin=8,
            competitor_price=p_comp,
            base_daily_demand=3.8,
            elasticity=1.7
        )

        # 1. Capacity check
        self.assertLessEqual(
            res_pulp["total_rooms_allocated"],
            c_rem + 1e-4,
            "Total allocated rooms must not exceed remaining capacity."
        )

        # 2. Competitor Price Index Hard Constraint Check (0.88 <= CPI <= 1.25)
        if res_pulp["total_rooms_allocated"] > 0:
            self.assertGreaterEqual(
                res_pulp["competitor_price_index"],
                self.lp_solver.min_cpi - 1e-3,
                "Realized portfolio price index must satisfy minimum competitor floor."
            )
            self.assertLessEqual(
                res_pulp["competitor_price_index"],
                self.lp_solver.max_cpi + 1e-3,
                "Realized portfolio price index must not exceed maximum competitor ceiling."
            )

        # 3. Dual / Shadow price check
        self.assertGreaterEqual(
            res_pulp["shadow_price_capacity"],
            0.0,
            "Capacity shadow price must be non-negative."
        )

    def test_lp_pulp_and_scipy_consistency(self):
        """PuLP and SciPy backends should yield equivalent optimal objective values."""
        res_pulp = self.lp_solver.solve(
            remaining_capacity=30,
            days_to_checkin=12,
            competitor_price=145.0,
            base_daily_demand=4.0,
            elasticity=1.8,
            backend="pulp"
        )
        res_scipy = self.lp_solver.solve(
            remaining_capacity=30,
            days_to_checkin=12,
            competitor_price=145.0,
            base_daily_demand=4.0,
            elasticity=1.8,
            backend="scipy"
        )

        self.assertAlmostEqual(
            res_pulp["total_revenue"],
            res_scipy["total_revenue"],
            delta=1.0,
            msg="PuLP and SciPy should produce matching optimal revenues."
        )
        self.assertAlmostEqual(
            res_pulp["total_rooms_allocated"],
            res_scipy["total_rooms_allocated"],
            delta=0.5
        )

    # -------------------------------------------------------------------------
    # 5. Full Simulation Loop & Trigger Mechanism Tests
    # -------------------------------------------------------------------------
    def test_full_simulation_pipeline(self):
        """Test complete 30-day simulation loop, occupancy updates, and performance metrics."""
        engine = HotelPricingEngine(
            total_rooms=50,
            baseline_fixed_price=135.0,
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

