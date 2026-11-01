"""
data_generator.py
Synthetic Market Feed Generator for Hotel Room Pricing Engine (INR / Indian Rupees).
Generates realistic, offline booking horizon market conditions without external APIs.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd


def generate_market_conditions(
    output_path: str = "data/market_conditions.csv",
    total_days: int = 30,
    total_rooms: int = 100,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic market condition feeds over a booking horizon in Indian Rupees (₹).

    Required columns:
      - days_to_checkin: Horizon countdown from T down to 1.
      - current_occupancy: Fraction of rooms currently booked (0.0 to 1.0).
      - competitor_avg_price: Average prevailing competitor rate (₹).
      - price_elasticity_factor: Price sensitivity of bookers at that lead time.
      - base_demand_rate: Base Poisson arrival intensity (arrivals/day).

    Dynamics simulated:
      1. Booking Build-up: Occupancy increases as check-in nears following an S-curve.
      2. Competitor Pricing: Competitors increase rates closer to check-in due to scarcity (₹4,800 to ₹9,200).
      3. Elasticity Compression: Early bookers (leisure) are price sensitive (high elasticity);
         late bookers (business/unplanned) are price inelastic (low elasticity).
      4. Demand Acceleration: Booking inquiries peak in the 2-week to 3-day lead window.
    """
    np.random.seed(random_seed)
    days = np.arange(total_days, 0, -1)  # 30 down to 1

    # 1. Base occupancy progression (Logistic S-curve with noise)
    # Starts around 15% occupancy at Day 30, reaches ~80-88% near check-in
    normalized_t = (total_days - days) / total_days  # 0.0 at t=30, ~1.0 at t=1
    s_curve = 0.15 + 0.70 / (1.0 + np.exp(-7.0 * (normalized_t - 0.45)))
    noise_occ = np.random.normal(0, 0.015, size=len(days))
    occupancy = np.clip(s_curve + noise_occ, 0.10, 0.95)
    occupancy = np.round(occupancy, 3)

    # 2. Competitor pricing dynamics in INR (₹4,800 baseline climbing to ~₹9,200 near check-in)
    comp_base = 4800.0
    comp_trend = 4200.0 * (normalized_t ** 1.3)
    comp_noise = np.random.normal(0, 120.0, size=len(days))
    competitor_price = np.round(comp_base + comp_trend + comp_noise, 2)

    # 3. Price elasticity factor
    # Early leisure travelers have higher price sensitivity (~2.4),
    # while last-minute business bookers are less sensitive (~1.1)
    elasticity_base = 2.4 - 1.25 * (normalized_t ** 0.8)
    elasticity_noise = np.random.normal(0, 0.04, size=len(days))
    price_elasticity = np.clip(np.round(elasticity_base + elasticity_noise, 2), 1.0, 3.0)

    # 4. Base arrival rate (Poisson intensity lambda_0)
    demand_curve = 2.0 + 2.5 * np.sin(np.pi * normalized_t) + 0.5 * normalized_t
    demand_noise = np.random.normal(0, 0.15, size=len(days))
    base_demand = np.clip(np.round(demand_curve + demand_noise, 2), 1.0, 6.0)

    df = pd.DataFrame({
        "days_to_checkin": days,
        "current_occupancy": occupancy,
        "competitor_avg_price": competitor_price,
        "price_elasticity_factor": price_elasticity,
        "base_demand_rate": base_demand
    })

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Data Generator] Successfully generated {len(df)} market records in INR (₹) at '{output_path}'.")
    return df


if __name__ == "__main__":
    generate_market_conditions()
