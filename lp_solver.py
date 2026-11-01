"""
lp_solver.py
Linear Programming (Network Revenue Management) Solver.
Formulates deterministic inventory allocation across customer segments
with hard competitor price index constraints and shadow price (bid price) extraction.
Supports both PuLP and SciPy HiGHS backends.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from typing import Dict, Any, Optional
import pulp
from scipy.optimize import linprog
from src.demand_model import HotelDemandModel


class LinearProgrammingRevenueSolver:
    """
    Deterministic Linear Programming (DLP) Network Revenue Management Solver.

    Decision Variables:
      x_j >= 0: Room inventory allocated to customer segment j
                (Budget, Standard, Premium)

    Objective:
      Maximize Total Revenue:
        max sum_{j} p_j * x_j

    Subject to:
      1. Remaining Capacity Constraint:
         sum_{j} x_j <= c_remaining
         Dual Variable (Shadow Price) mu_capacity = LP Bid Price (Marginal value of capacity).

      2. Segment Demand Upper Bounds:
         0 <= x_j <= D_j(p_j, p_comp, t)

      3. Hard Competitor Price Index (CPI) Constraints:
         theta_min <= (sum_j p_j * x_j) / (p_comp * sum_j x_j) <= theta_max
         Linearized:
           sum_j (p_j - theta_min * p_comp) * x_j >= 0   (Floor parity constraint)
           sum_j (p_j - theta_max * p_comp) * x_j <= 0   (Ceiling competitive constraint)
    """

    def __init__(
        self,
        demand_model: Optional[HotelDemandModel] = None,
        min_competitor_index: float = 0.88,
        max_competitor_index: float = 1.25
    ):
        """
        Parameters:
            demand_model: Instance of HotelDemandModel.
            min_competitor_index: Lower bound on portfolio average price relative to competitor.
            max_competitor_index: Upper bound on portfolio average price relative to competitor.
        """
        self.demand_model = demand_model or HotelDemandModel()
        self.min_cpi = min_competitor_index
        self.max_cpi = max_competitor_index

    def solve(
        self,
        remaining_capacity: int,
        days_to_checkin: int,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float,
        backend: str = "pulp"
    ) -> Dict[str, Any]:
        """
        Solve deterministic capacity allocation LP using either PuLP or SciPy.
        """
        if backend.lower() == "scipy":
            return self.solve_scipy(
                remaining_capacity=remaining_capacity,
                days_to_checkin=days_to_checkin,
                competitor_price=competitor_price,
                base_daily_demand=base_daily_demand,
                elasticity=elasticity
            )
        else:
            return self.solve_pulp(
                remaining_capacity=remaining_capacity,
                days_to_checkin=days_to_checkin,
                competitor_price=competitor_price,
                base_daily_demand=base_daily_demand,
                elasticity=elasticity
            )

    def solve_pulp(
        self,
        remaining_capacity: int,
        days_to_checkin: int,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float
    ) -> Dict[str, Any]:
        """
        Solve deterministic capacity allocation LP using PuLP with HiGHS / CBC solver.
        """
        c_rem = max(0, remaining_capacity)
        if c_rem == 0:
            return {
                "status": "Optimal",
                "total_revenue": 0.0,
                "allocation": {"Budget": 0.0, "Standard": 0.0, "Premium": 0.0},
                "total_rooms_allocated": 0.0,
                "weighted_avg_price": 0.0,
                "shadow_price_capacity": 0.0,
                "recommended_dynamic_price": 0.0,
                "competitor_price_index": 0.0,
                "solver_backend": "PuLP"
            }

        # Segment demands and prices
        segment_data = self.demand_model.expected_segment_demands(
            days_remaining=days_to_checkin,
            competitor_price=competitor_price,
            base_daily_demand=base_daily_demand,
            elasticity=elasticity
        )

        segments = list(segment_data.keys())
        prices = {s: segment_data[s]["price"] for s in segments}
        demands = {s: segment_data[s]["expected_demand"] for s in segments}

        prob = pulp.LpProblem("Hotel_Revenue_Management_NRM", pulp.LpMaximize)

        # Handle PuLP versions (PuLP 4.x add_variable vs classic LpVariable)
        x_vars = {}
        for s in segments:
            if hasattr(prob, "add_variable"):
                x_vars[s] = prob.add_variable(
                    name=f"x_{s}",
                    lowBound=0.0,
                    upBound=demands[s],
                    cat="Continuous"
                )
            else:
                x_vars[s] = pulp.LpVariable(
                    name=f"x_{s}",
                    lowBound=0.0,
                    upBound=demands[s],
                    cat=pulp.LpContinuous
                )

        # Objective Function: Total Revenue = sum (p_j * x_j)
        prob += pulp.lpSum([prices[s] * x_vars[s] for s in segments]), "Total_Expected_Revenue"

        # 1. Capacity Constraint: sum x_j <= c_remaining
        prob += pulp.lpSum([x_vars[s] for s in segments]) <= c_rem, "Capacity_Constraint"

        # 2. Competitor Price Index Lower Bound (Hard Constraint):
        # sum (p_j - min_cpi * p_comp) * x_j >= 0
        min_coeff = {s: prices[s] - self.min_cpi * competitor_price for s in segments}
        prob += pulp.lpSum([min_coeff[s] * x_vars[s] for s in segments]) >= 0, "Min_Competitor_Price_Index"

        # 3. Competitor Price Index Upper Bound (Hard Constraint):
        # sum (p_j - max_cpi * p_comp) * x_j <= 0
        max_coeff = {s: prices[s] - self.max_cpi * competitor_price for s in segments}
        prob += pulp.lpSum([max_coeff[s] * x_vars[s] for s in segments]) <= 0, "Max_Competitor_Price_Index"

        # Solver selection
        solver = None
        if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
            solver = pulp.getSolver("HiGHS", msg=False)
        elif "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True):
            solver = pulp.PULP_CBC_CMD(msg=False)

        if solver is not None:
            res = prob.solve(solver)
        else:
            res = prob.solve()

        # Extract variable values
        alloc = {s: round(float(x_vars[s].varValue or 0.0), 2) for s in segments}
        tot_allocated = sum(alloc.values())

        # Extract objective
        if hasattr(res, "objective") and res.objective is not None:
            tot_rev = round(float(res.objective), 2)
        elif hasattr(prob, "objective") and prob.objective is not None:
            tot_rev = round(float(prob.objective.value() or 0.0), 2)
        else:
            tot_rev = round(sum(prices[s] * alloc[s] for s in segments), 2)

        # Extract shadow price of capacity constraint
        shadow_price = 0.0
        try:
            if callable(getattr(prob, "constraints", None)):
                for c in prob.constraints():
                    if c.name == "Capacity_Constraint" and c.pi is not None:
                        shadow_price = round(abs(float(c.pi)), 2)
                        break
            elif isinstance(prob.constraints, dict) and "Capacity_Constraint" in prob.constraints:
                pi_val = prob.constraints["Capacity_Constraint"].pi
                if pi_val is not None:
                    shadow_price = round(abs(float(pi_val)), 2)
        except Exception:
            shadow_price = 0.0

        status_str = getattr(res, "status_str", "Optimal")

        # Weighted average rate across allocated segments
        if tot_allocated > 0:
            weighted_avg_p = round(tot_rev / tot_allocated, 2)
            cpi_actual = round(weighted_avg_p / competitor_price, 3)
        else:
            weighted_avg_p = round(competitor_price, 2)
            cpi_actual = 1.0

        # In RM bid pricing: price to offer = max of standard segment price or capacity opportunity cost
        recommended_price = max(prices["Standard"], shadow_price) if shadow_price > 0 else weighted_avg_p

        return {
            "status": status_str,
            "total_revenue": tot_rev,
            "allocation": alloc,
            "total_rooms_allocated": round(tot_allocated, 2),
            "weighted_avg_price": weighted_avg_p,
            "shadow_price_capacity": shadow_price,
            "recommended_dynamic_price": round(recommended_price, 2),
            "competitor_price_index": cpi_actual,
            "segment_demands": demands,
            "segment_prices": prices,
            "solver_backend": "PuLP (HiGHS)"
        }

    def solve_scipy(
        self,
        remaining_capacity: int,
        days_to_checkin: int,
        competitor_price: float,
        base_daily_demand: float,
        elasticity: float
    ) -> Dict[str, Any]:
        """
        Solve deterministic capacity allocation LP using SciPy linprog (HiGHS).
        """
        c_rem = max(0, remaining_capacity)
        if c_rem == 0:
            return {
                "status": "Optimal",
                "total_revenue": 0.0,
                "allocation": {"Budget": 0.0, "Standard": 0.0, "Premium": 0.0},
                "total_rooms_allocated": 0.0,
                "weighted_avg_price": 0.0,
                "shadow_price_capacity": 0.0,
                "recommended_dynamic_price": 0.0,
                "competitor_price_index": 0.0,
                "solver_backend": "SciPy-HiGHS"
            }

        segment_data = self.demand_model.expected_segment_demands(
            days_remaining=days_to_checkin,
            competitor_price=competitor_price,
            base_daily_demand=base_daily_demand,
            elasticity=elasticity
        )

        segments = list(segment_data.keys())
        prices = np.array([segment_data[s]["price"] for s in segments])
        demands = np.array([segment_data[s]["expected_demand"] for s in segments])

        # SciPy linprog minimizes c^T x -> to maximize revenue, minimize -prices^T x
        c = -prices

        # Inequality constraints: A_ub * x <= b_ub
        # 1. Capacity: 1*x_0 + 1*x_1 + 1*x_2 <= c_rem
        # 2. Min CPI: -sum (p_j - theta_min * p_comp) x_j <= 0
        # 3. Max CPI: sum (p_j - theta_max * p_comp) x_j <= 0
        A_ub = [
            [1.0, 1.0, 1.0],
            list(-(prices - self.min_cpi * competitor_price)),
            list(prices - self.max_cpi * competitor_price)
        ]
        b_ub = [float(c_rem), 0.0, 0.0]
        bounds = [(0.0, float(demands[i])) for i in range(len(segments))]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            return {
                "status": "Infeasible / Failed",
                "total_revenue": 0.0,
                "allocation": {s: 0.0 for s in segments},
                "total_rooms_allocated": 0.0,
                "weighted_avg_price": 0.0,
                "shadow_price_capacity": 0.0,
                "recommended_dynamic_price": 0.0,
                "competitor_price_index": 0.0,
                "solver_backend": "SciPy-HiGHS"
            }

        x_vals = res.x
        alloc = {segments[i]: round(float(x_vals[i]), 2) for i in range(len(segments))}
        tot_allocated = sum(alloc.values())
        tot_rev = round(-float(res.fun), 2)

        # Capacity shadow price from SciPy duals
        shadow_price = 0.0
        if hasattr(res, "ineqlin") and res.ineqlin is not None and hasattr(res.ineqlin, "marginals"):
            shadow_price = round(abs(float(res.ineqlin.marginals[0])), 2)

        if tot_allocated > 0:
            weighted_avg_p = round(tot_rev / tot_allocated, 2)
            cpi_actual = round(weighted_avg_p / competitor_price, 3)
        else:
            weighted_avg_p = round(competitor_price, 2)
            cpi_actual = 1.0

        p_standard = float(prices[segments.index("Standard")])
        recommended_price = max(p_standard, shadow_price) if shadow_price > 0 else weighted_avg_p

        return {
            "status": "Optimal",
            "total_revenue": tot_rev,
            "allocation": alloc,
            "total_rooms_allocated": round(tot_allocated, 2),
            "weighted_avg_price": weighted_avg_p,
            "shadow_price_capacity": shadow_price,
            "recommended_dynamic_price": round(recommended_price, 2),
            "competitor_price_index": cpi_actual,
            "segment_demands": {segments[i]: round(float(demands[i]), 2) for i in range(len(segments))},
            "segment_prices": {segments[i]: round(float(prices[i]), 2) for i in range(len(segments))},
            "solver_backend": "SciPy-HiGHS"
        }
