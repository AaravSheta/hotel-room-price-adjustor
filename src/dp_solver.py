"""
dp_solver.py
Dynamic Programming (MDP) Solver for Hotel Revenue Management (INR / Indian Rupees).
Implements the Bellman Optimality Equation over a finite booking horizon
using Poisson batch arrival expectations and competitor-anchored action tiers.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import poisson
from typing import List, Dict, Tuple, Optional
from src.demand_model import HotelDemandModel


class DynamicProgrammingPricingSolver:
    """
    Finite-Horizon Markov Decision Process (MDP) solver for dynamic room pricing in INR (₹).

    State:
      (c, t) where:
        c = number of unsold rooms (inventory capacity, 0 <= c <= total_capacity)
        t = days remaining until check-in (0 <= t <= total_days)

    Action:
      p in P(p_comp) = { alpha_k * p_comp | alpha_k in candidate_multipliers }
      Price tiers dynamically anchor to the prevailing competitor reference price,
      ranging from deep discount (to prevent spoilage during surplus) to premium surge (during scarcity).

    Bellman Equation (with Poisson Arrival Distribution):
      In stage t, customer arrivals k follow Poisson(lambda(p, p_comp)):
      V_t(c) = max_{p in P} sum_{k=0}^{K_max} P(k | lambda(p)) * [ p * min(c, k) + V_{t-1}(max(0, c - k)) ]

      Marginal Value / Capacity Bid Price:
      Delta V_t(c) = V_t(c) - V_t(c - 1)
    """

    def __init__(
        self,
        total_capacity: int = 100,
        demand_model: Optional[HotelDemandModel] = None,
        price_multipliers: Optional[List[float]] = None
    ):
        """
        Initialize the DP Pricing Solver.

        Parameters:
            total_capacity: Maximum hotel room inventory.
            demand_model: Instance of HotelDemandModel.
            price_multipliers: Ratio tiers relative to competitor benchmark.
        """
        self.capacity = total_capacity
        self.demand_model = demand_model or HotelDemandModel()

        if price_multipliers is None:
            # Competitor-relative pricing corridor from discount to peak surge
            self.price_multipliers = np.array([
                0.65, 0.72, 0.80, 0.88, 0.95,
                1.00, 1.06, 1.15, 1.25, 1.40, 1.65
            ], dtype=float)
        else:
            self.price_multipliers = np.array(sorted(price_multipliers), dtype=float)

    def solve_bellman(
        self,
        remaining_days: int,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float,
        k_max: int = 12
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the Bellman equation via vectorized backward induction over remaining days.

        Parameters:
            remaining_days: Total days remaining until check-in (t = 1 to remaining_days).
            competitor_price: Current competitor average benchmark rate (₹).
            base_daily_demand: Base Poisson arrival intensity (arrivals/day).
            elasticity: Price elasticity of demand.
            k_max: Upper summation limit for daily Poisson arrivals.

        Returns:
            V: Expected value matrix of shape (capacity + 1, remaining_days + 1)
            policy: Optimal price policy matrix of shape (capacity + 1, remaining_days + 1)
            bid_prices: Marginal capacity value matrix Delta V of shape (capacity + 1, remaining_days + 1)
            prices: Discrete candidate prices offered at this competitor benchmark
        """
        C = self.capacity
        T = max(1, remaining_days)

        # Candidate price tiers anchored to competitor benchmark (rounded to nearest ₹50)
        prices = np.round(competitor_price * self.price_multipliers / 50.0) * 50.0

        V = np.zeros((C + 1, T + 1), dtype=float)
        policy = np.zeros((C + 1, T + 1), dtype=float)
        bid_prices = np.zeros((C + 1, T + 1), dtype=float)

        k_vals = np.arange(0, k_max + 1)  # shape: (k_max + 1,)
        c_range = np.arange(0, C + 1)     # shape: (C + 1,)

        c_grid = c_range[:, np.newaxis]   # shape: (C + 1, 1)
        k_grid = k_vals[np.newaxis, :]     # shape: (1, k_max + 1)

        # Precompute sales and next capacity grids for all (c, k) combinations
        sales_grid = np.minimum(c_grid, k_grid)         # shape: (C + 1, k_max + 1)
        next_c_grid = np.maximum(0, c_grid - k_grid)    # shape: (C + 1, k_max + 1)

        # Precompute arrival probabilities for each candidate price
        prob_matrix = []
        for p in prices:
            price_ratio = max(0.1, p / competitor_price)
            lam = max(0.05, base_daily_demand * (price_ratio ** (-elasticity)))
            pmf = poisson.pmf(k_vals, lam)
            tail = max(0.0, 1.0 - np.sum(pmf))
            pmf[-1] += tail
            prob_matrix.append(pmf)
        prob_matrix = np.array(prob_matrix)  # shape: (num_prices, k_max + 1)

        # Backward Induction from t = 1 to T
        for t in range(1, T + 1):
            q_candidates = []
            future_v = V[next_c_grid, t - 1]  # shape: (C + 1, k_max + 1)

            for i, p in enumerate(prices):
                probs = prob_matrix[i][np.newaxis, :]  # shape: (1, k_max + 1)
                immed_rev = sales_grid * p             # shape: (C + 1, k_max + 1)
                total_exp = np.sum(probs * (immed_rev + future_v), axis=1)  # shape: (C + 1,)
                q_candidates.append(total_exp)

            q_matrix = np.array(q_candidates)  # shape: (num_prices, C + 1)
            best_idx = np.argmax(q_matrix, axis=0)  # shape: (C + 1,)

            V[:, t] = np.max(q_matrix, axis=0)
            policy[:, t] = prices[best_idx]
            bid_prices[1:, t] = V[1:, t] - V[:-1, t]

        return V, policy, bid_prices, prices

    def get_optimal_price(
        self,
        current_unsold_rooms: int,
        days_to_checkin: int,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float
    ) -> Dict[str, float]:
        """
        Compute real-time optimal price for the exact current state (c, t).

        Returns:
            Dictionary with optimal_price, expected_future_revenue,
            marginal_bid_price, and booking_probability in INR (₹).
        """
        c = max(0, min(current_unsold_rooms, self.capacity))
        days = max(1, days_to_checkin)

        if c == 0:
            return {
                "optimal_price": 0.0,
                "expected_future_revenue": 0.0,
                "marginal_bid_price": 0.0,
                "booking_probability": 0.0,
                "periods_remaining": days,
                "competitor_price_ratio": 0.0
            }

        V, policy, bid_prices, _ = self.solve_bellman(
            remaining_days=days,
            competitor_price=competitor_price,
            base_daily_demand=base_daily_demand,
            elasticity=elasticity
        )

        opt_p = float(policy[c, days])
        exp_rev = float(V[c, days])
        bid_p = float(bid_prices[c, days])

        # Booking probability per decision epoch (consistent with demand_model, baseline, and LP)
        prob_epoch = self.demand_model.booking_probability_poisson(
            price=opt_p,
            competitor_price=competitor_price,
            base_daily_demand=base_daily_demand,
            elasticity=elasticity
        )

        # Full-day booking probability (P >= 1 arrival in 24 hours)
        price_ratio = opt_p / competitor_price
        lam_day = base_daily_demand * (price_ratio ** (-elasticity))
        prob_day = float(np.clip(1.0 - np.exp(-lam_day), 0.0, 1.0))

        return {
            "optimal_price": float(round(opt_p, 2)),
            "expected_future_revenue": float(round(exp_rev, 2)),
            "marginal_bid_price": float(round(bid_p, 2)),
            "booking_probability": float(round(prob_epoch, 4)),
            "booking_probability_daily": float(round(prob_day, 4)),
            "periods_remaining": days,
            "competitor_price_ratio": float(round(price_ratio, 3))
        }
