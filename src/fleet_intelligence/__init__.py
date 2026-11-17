"""
Fleet Intelligence Package
"""

from .fleet_health import calculate_fleet_health
from .risk_ranking import rank_machines_by_risk
from .fleet_analytics import generate_fleet_analytics

__all__ = [
    "calculate_fleet_health",
    "rank_machines_by_risk",
    "generate_fleet_analytics",
]