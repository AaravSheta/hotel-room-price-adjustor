"""
dp_solver.py
Dynamic Programming (MDP) Solver for Hotel Revenue Management.
Implements the Bellman Optimality Equation over a finite booking horizon.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.demand_model import HotelDemandModel


class DynamicProgrammingPricingSolver:
    """
    Finite-Horizon Markov Decision Process (MDP) solver for dynamic room pricing.

    State:
      (c, t) where:
        c = number of unsold rooms (inventory capacity, 0 <= c <= total_capacity)
        t = decision periods remaining until check-in (0 <= t <= total_periods)

    Action:
      p in P (discrete price tiers)

    Reward & Transition:
      With probability P(booking | p, p_comp), transition to (c - 1, t - 1) with revenue p.
      With probability 1 - P(booking | p, p_comp), transition to (c, t - 1) with revenue 0.

    Bellman Equation:
      V_t(c) = max_{p in P} { P(p, p_comp) * [p + V_{t-1}(c - 1)] + (1 - P(p, p_comp)) * V_{t-1}(c) }
             = V_{t-1}(c) + max_{p in P} { P(p, p_comp) * [p - Delta V_{t-1}(c)] }
      where Delta V_{t-1}(c) = V_{t-1}(c) - V_{t-1}(c - 1) is the Marginal Value / Bid Price.
    """

    def __init__(
        self,
        total_capacity: int = 100,
        demand_model: Optional[HotelDemandModel] = None,
        price_tiers: Optional[List[float]] = None
    ):
        """
        Initialize the DP Pricing Solver.

        Parameters:
            total_capacity: Maximum hotel room inventory.
            demand_model: Instance of HotelDemandModel.
            price_tiers: Feasible discrete price levels ($). If None, defaults to standard tiers.
        """
        self.capacity = total_capacity
        self.demand_model = demand_model or HotelDemandModel()

        if price_tiers is None:
            # Diverse pricing spectrum from budget discount to peak surge
            self.price_tiers = np.array([
                85.0, 100.0, 115.0, 130.0, 145.0, 160.0,
                175.0, 195.0, 215.0, 240.0, 270.0, 310.0
            ], dtype=float)
        else:
            self.price_tiers = np.array(sorted(price_tiers), dtype=float)

    def solve_bellman(
        self,
        remaining_periods: int,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the Bellman equation via backward induction.

        Parameters:
            remaining_periods: Total discrete stages remaining (t = 1 to remaining_periods).
            competitor_price: Current competitor average benchmark rate ($).
            base_daily_demand: Base Poisson arrival intensity (arrivals/day).
            elasticity: Price elasticity of demand.

        Returns:
            V: Value function matrix of shape (capacity + 1, remaining_periods + 1)
            policy: Optimal price policy matrix of shape (capacity + 1, remaining_periods + 1)
            bid_prices: Marginal capacity value matrix Delta V of shape (capacity + 1, remaining_periods + 1)
        """
        C = self.capacity
        T = remaining_periods

        # V[c, t] stores expected revenue with c rooms remaining and t periods left
        V = np.zeros((C + 1, T + 1), dtype=float)
        # policy[c, t] stores optimal price tier to offer
        policy = np.zeros((C + 1, T + 1), dtype=float)
        # bid_prices[c, t] stores V_t(c) - V_t(c - 1)
        bid_prices = np.zeros((C + 1, T + 1), dtype=float)

        # Precompute booking arrival probabilities for all price tiers given current market conditions
        probs = np.array([
            self.demand_model.booking_probability_poisson(
                price=p,
                competitor_price=competitor_price,
                base_daily_demand=base_daily_demand,
                elasticity=elasticity
            )
            for p in self.price_tiers
        ])  # Shape: (len(price_tiers),)

        # Backward Induction from t = 1 to T
        for t in range(1, T + 1):
            # For c = 0, no rooms to sell -> V[0, t] = 0, policy = min price
            policy[0, t] = self.price_tiers[0]

            # Vectorized over all positive room capacities c in [1, C]
            # V_{t-1}(c) has shape (C,)
            v_prev_same = V[1:C + 1, t - 1]      # V_{t-1}(c)
            v_prev_minus1 = V[0:C, t - 1]        # V_{t-1}(c - 1)

            # Opportunity cost (bid price) of selling 1 room:
            # delta_V = V_{t-1}(c) - V_{t-1}(c - 1)
            delta_V = v_prev_same - v_prev_minus1  # shape: (C,)

            # For each price p in price_tiers:
            # Expected incremental gain: prob * (p - delta_V)
            # Shapes:
            # probs: (num_prices, 1)
            # price_tiers: (num_prices, 1)
            # delta_V: (1, C)
            probs_2d = probs[:, np.newaxis]             # (P, 1)
            prices_2d = self.price_tiers[:, np.newaxis]  # (P, 1)
            delta_v_2d = delta_V[np.newaxis, :]         # (1, C)

            # Net expected revenue if choosing price p with capacity c:
            # immediate expected revenue + continuation value:
            # Q_t(c, p) = probs * (p + v_prev_minus1) + (1 - probs) * v_prev_same
            #           = v_prev_same + probs * (p - delta_V)
            q_values = v_prev_same + probs_2d * (prices_2d - delta_v_2d)  # shape: (P, C)

            # Maximize over price tiers (axis 0)
            best_idx = np.argmax(q_values, axis=0)       # shape: (C,)
            best_values = np.max(q_values, axis=0)       # shape: (C,)

            V[1:C + 1, t] = best_values
            policy[1:C + 1, t] = self.price_tiers[best_idx]

            # Record bid prices at stage t
            bid_prices[1:C + 1, t] = V[1:C + 1, t] - V[0:C, t]

        return V, policy, bid_prices

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
            marginal_bid_price, and booking_probability.
        """
        c = max(0, min(current_unsold_rooms, self.capacity))
        if c == 0:
            return {
                "optimal_price": 0.0,
                "expected_future_revenue": 0.0,
                "marginal_bid_price": 0.0,
                "booking_probability": 0.0
            }

        # Convert remaining days into decision epochs
        total_periods = max(1, int(days_to_checkin * self.demand_model.daily_steps))

        V, policy, bid_prices = self.solve_bellman(
            remaining_periods=total_periods,
            competitor_price=competitor_price,
            base_daily_demand=base_daily_demand,
            elasticity=elasticity
        )

        opt_p = policy[c, total_periods]
        exp_rev = V[c, total_periods]
        bid_p = bid_prices[c, total_periods]
        prob = self.demand_model.booking_probability_poisson(
            price=opt_p,
            competitor_price=competitor_price,
            base_daily_demand=base_daily_demand,
            elasticity=elasticity
        )

        return {
            "optimal_price": float(round(opt_p, 2)),
            "expected_future_revenue": float(round(exp_rev, 2)),
            "marginal_bid_price": float(round(bid_p, 2)),
            "booking_probability": float(round(prob, 4)),
            "periods_remaining": total_periods
        }

