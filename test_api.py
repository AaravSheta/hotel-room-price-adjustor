"""
test_api.py
Unit tests for the FastAPI dynamic pricing models and endpoints (INR ₹).
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
from server import optimize_price, PricingRequest, health_check, get_market_feed, serve_frontend_root


class TestPricingAPI(unittest.TestCase):

    def test_health_check(self):
        res = health_check()
        self.assertEqual(res["status"], "healthy")
        self.assertIn("solvers", res)

    def test_market_feed(self):
        feed = get_market_feed()
        self.assertIsInstance(feed, list)
        self.assertEqual(len(feed), 30)
        self.assertIn("competitor_avg_price", feed[0])
        self.assertGreater(feed[0]["competitor_avg_price"], 2000.0)

    def test_optimize_price_calculation(self):
        req = PricingRequest(
            days_to_checkin=10,
            current_occupancy=75.0,  # 75%
            competitor_avg_price=6500.0,
            base_price=5500.0,
            total_rooms=100
        )
        res = optimize_price(req)
        self.assertEqual(res.status, "success")

        # Verify inputs
        self.assertEqual(res.inputs["days_to_checkin"], 10)
        self.assertEqual(res.inputs["unsold_rooms"], 25)

        # Verify baseline, DP, LP outputs
        self.assertGreater(res.baseline["price"], 2000)
        self.assertGreater(res.dynamic_programming["price"], 2000)
        self.assertGreater(res.linear_programming["price"], 2000)

        # Capacity bid price & shadow price
        self.assertGreaterEqual(res.dynamic_programming["marginal_bid_price"], 0)
        self.assertGreaterEqual(res.linear_programming["shadow_price_capacity"], 0)

        # Allocation checks
        alloc = res.linear_programming["allocation"]
        self.assertIn("Budget", alloc)
        self.assertIn("Standard", alloc)
        self.assertIn("Premium", alloc)
        self.assertLessEqual(sum(alloc.values()), 25.1)

    def test_fractional_occupancy(self):
        req = PricingRequest(
            days_to_checkin=3,
            current_occupancy=0.85,  # fractional 85%
            competitor_avg_price=8500.0,
            base_price=5500.0
        )
        res = optimize_price(req)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.inputs["current_occupancy_pct"], 85.0)
        self.assertEqual(res.inputs["unsold_rooms"], 15)

    def test_frontend_serving(self):
        res = serve_frontend_root()
        self.assertEqual(res.status_code, 200)
        self.assertIn("Hotel Revenue Management Engine", res.body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
