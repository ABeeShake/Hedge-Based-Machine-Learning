"""fast_engine/__init__.py — High-Performance Online Forecasting & Hedging Engine.

This package provides a vectorized, low-overhead simulation architecture
designed to replace the multi-month PyTorch Lightning/sequential simulation loop.
"""

from .models import (
    FastNODE,
    FastNHITS,
    FastXGBoost,
    FastStatsForecaster,
)
from .simulation import FastRollingSimulation, run_cohort
from .hedge import FastOnlineHedge

__all__ = [
    "FastNODE",
    "FastNHITS",
    "FastXGBoost",
    "FastStatsForecaster",
    "FastRollingSimulation",
    "FastOnlineHedge",
    "run_cohort",
]
