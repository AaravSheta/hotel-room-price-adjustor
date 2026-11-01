"""
demand_model.py
Probabilistic Demand and Customer Arrival Modeling for Hotel Revenue Management.
Models Poisson arrival processes, price elasticity, and competitor price sensitivity.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np


class HotelDemandModel:
    """
    Probabilistic demand model for hotel room booking inquiries.

    Combines:
      - Competitor relative pricing (Price Index = p / p_comp)
      - Constant/lead-time dependent price elasticity (epsilon)
      - Non-homogeneous Poisson arrival processes
      - Multinomial discrete choice (MNL) formulation
    """

    def __init__(self, time_step_hours: float = 4.0):
        """
        Parameters:
            time_step_hours: Duration of a single decision period in hours.
                             Default is 4 hours (6 decision steps per day),
                             ensuring at most 1 arrival per discrete interval.
        """
        self.time_step_hours = time_step_hours
        self.daily_steps = 24.0 / time_step_hours

    def calculate_arrival_rate(
        self,
        price: float,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float
    ) -> float:
        """
        Calculate the adjusted Poisson arrival rate (lambda) for a single decision period.

        Formula:
            lambda_period = (lambda_base / daily_steps) * (price / competitor_price) ** (-elasticity)

        Parameters:
            price: Proposed hotel room rate ($).
            competitor_price: Market competitor average rate ($).
            base_daily_demand: Expected arrivals per day at price parity (p = p_comp).
            elasticity: Price elasticity of demand (positive float, e.g. 1.2 to 2.5).

        Returns:
            Adjusted Poisson arrival rate for the decision interval.
        """
        if price <= 0 or competitor_price <= 0:
            raise ValueError("Prices must be strictly positive.")

        price_ratio = price / competitor_price
        # Demand elasticity response: (p / p_comp)^(-elasticity)
        elasticity_multiplier = price_ratio ** (-elasticity)

        # Base arrival rate scaled to decision interval
        base_period_demand = base_daily_demand / self.daily_steps
        adjusted_rate = base_period_demand * elasticity_multiplier

        return float(adjusted_rate)

    def booking_probability_poisson(
        self,
        price: float,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float
    ) -> float:
        """
        Calculate the probability of at least one booking arrival during the decision period
        using the standard Poisson arrival formulation:
            P(Booking >= 1) = 1 - exp(-lambda_period)

        Returns:
            Probability value in [0.0, 1.0].
        """
        arr_rate = self.calculate_arrival_rate(
            price=price,
            competitor_price=competitor_price,
            base_daily_demand=base_daily_demand,
            elasticity=elasticity
        )
        # Poisson probability of at least one arrival
        prob = 1.0 - np.exp(-arr_rate)
        return float(np.clip(prob, 0.0, 1.0))

    def booking_probability_logit(
        self,
        price: float,
        competitor_price: float,
        quality_delta: float = 0.0,
        outside_utility: float = -1.5,
        price_sensitivity: float = 0.035
    ) -> float:
        """
        Multinomial Logit (MNL) discrete choice model.
        Customer chooses between:
          1. Our Hotel (utility = V_our - beta * price)
          2. Competitor Hotel (utility = V_comp - beta * p_comp)
          3. Outside Option / No Purchase (utility = outside_utility)

        Returns:
          Probability of customer choosing our hotel.
        """
        u_our = quality_delta - price_sensitivity * price
        u_comp = -price_sensitivity * competitor_price
        u_out = outside_utility

        max_u = max(u_our, u_comp, u_out)
        exp_our = np.exp(u_our - max_u)
        exp_comp = np.exp(u_comp - max_u)
        exp_out = np.exp(u_out - max_u)

        prob_our = exp_our / (exp_our + exp_comp + exp_out)
        return float(prob_our)

    def expected_segment_demands(
        self,
        days_remaining: int,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float,
        segment_weights: dict = None
    ) -> dict:
        """
        Compute total expected demand across customer segments for the remaining horizon.
        Used by Network Revenue Management Linear Program.

        Customer Segments:
          - Budget (Leisure / Price Sensitive, high elasticity)
          - Standard (Transient / Normal sensitivity)
          - Premium (Business / Corporate, inelastic)
        """
        if segment_weights is None:
            segment_weights = {
                "Budget": {"share": 0.35, "price_mult": 0.80, "elasticity_mult": 1.30},
                "Standard": {"share": 0.45, "price_mult": 1.00, "elasticity_mult": 1.00},
                "Premium": {"share": 0.20, "price_mult": 1.35, "elasticity_mult": 0.70},
            }

        total_base_inquiries = base_daily_demand * days_remaining
        segment_demands = {}

        for seg, props in segment_weights.items():
            seg_price = competitor_price * props["price_mult"]
            seg_elasticity = elasticity * props["elasticity_mult"]
            price_ratio = seg_price / competitor_price
            demand_multiplier = price_ratio ** (-seg_elasticity)

            expected_d = total_base_inquiries * props["share"] * demand_multiplier
            segment_demands[seg] = {
                "price": round(seg_price, 2),
                "expected_demand": max(1.0, round(float(expected_d), 2)),
                "elasticity": round(seg_elasticity, 2)
            }

        return segment_demands

