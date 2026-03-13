"""
Global method registry — single source of truth for method keys and display names.

Every module that needs method names (experiment, visualization, paper tables, etc.)
should import from here instead of hardcoding strings.
"""

METHOD_REGISTRY = {
    "slm_manifold": {"display": "SLM-Manifold", "role": "proposed"},
    "bcd_slm":      {"display": "BCD-SLM",      "role": "proposed"},
    "slm_interior": {"display": "SLM-Interior",  "role": "baseline"},
    "slm_oracle":   {"display": "SLM-Oracle",    "role": "diagnostic"},
    "em":           {"display": "EM",            "role": "baseline"},
    "ecm":          {"display": "ECM",           "role": "baseline"},
}

# Simulation experiments (Section 6.1): all 6 methods
SIMULATION_METHODS = ["slm_manifold", "bcd_slm", "slm_interior", "slm_oracle", "em", "ecm"]

# Prediction experiments (Section 6.2): no SLM-Interior, add PLSR/Ridge
PREDICTION_METHODS = ["slm_manifold", "em", "plsr", "ridge"]

def display_name(key: str) -> str:
    """Return the paper-ready display name for a method key."""
    return METHOD_REGISTRY[key]["display"]
