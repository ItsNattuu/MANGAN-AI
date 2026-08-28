"""
mining_efficiency.py
----------------------
Layer 3, Part C: combine terrain, accessibility, infrastructure, and
constraints into a single Mining Efficiency Score (0-100).

IMPORTANT — the only Layer 2 dependency in this whole file is the
`prospectivity_score` argument to compute_mining_efficiency(). Until
Layer 2's model is ready, use get_mock_prospectivity_score() below so
you can build, test, and demo this file completely independently.
When Layer 2 is ready, replace that ONE function call — nothing else
in this file changes.
"""

import random
import numpy as np

from infrastructure_features import InfrastructureFeatureEngine
from land_constraints import LandConstraintEngine
from terrain_features import TerrainFeatureEngine  # reuse the same one Layer 2 built


# ---------------------------------------------------------------------
# TEMPORARY STAND-IN FOR LAYER 2 — delete this once the real model exists
# ---------------------------------------------------------------------
def get_mock_prospectivity_score(lon, lat, seed=None):
    """
    Returns a fake P(manganese) between 0 and 1, just so Layer 3's
    logic can be built and tested end-to-end without waiting on
    Layer 2. Replace calls to this with the real model's output:

        from predict_prospectivity import get_prospectivity_at_point
        score = get_prospectivity_at_point(lon, lat)
    """
    rng = random.Random(seed if seed is not None else f"{lon}_{lat}")
    return rng.uniform(0.3, 0.95)


# ---------------------------------------------------------------------
# Normalization helpers — convert raw distances/slopes into 0-1 "good-ness"
# scores where 1 = ideal for mining, 0 = worst case.
# ---------------------------------------------------------------------

def normalize_distance(distance_m, max_acceptable_m):
    """
    Closer = better. Returns 1.0 at distance=0, decaying to 0.0 at
    max_acceptable_m and beyond. max_acceptable_m is a judgment call —
    e.g. 20 km for roads, 50 km for railways — document your choices.
    """
    if np.isnan(distance_m):
        return 0.0
    score = 1.0 - (distance_m / max_acceptable_m)
    return float(np.clip(score, 0.0, 1.0))


def normalize_slope(slope_deg, ideal_max_deg=15.0, unusable_deg=35.0):
    """
    Flatter terrain = easier/cheaper mining. Below ideal_max_deg -> 1.0.
    Above unusable_deg -> 0.0. Linear falloff in between.
    """
    if np.isnan(slope_deg):
        return 0.0
    if slope_deg <= ideal_max_deg:
        return 1.0
    if slope_deg >= unusable_deg:
        return 0.0
    return float(1.0 - (slope_deg - ideal_max_deg) / (unusable_deg - ideal_max_deg))


# ---------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------

# Weights are a transparent, explainable prototype choice — NOT derived
# from real economic data. Say this explicitly in your report (see the
# earlier ChatGPT excerpt, point 16).
EFFICIENCY_WEIGHTS = {
    "prospectivity": 0.40,
    "accessibility": 0.20,
    "terrain": 0.15,
    "infrastructure": 0.15,
    "land_eligibility": 0.10,
}


def compute_mining_efficiency(
    lon, lat,
    infra_engine: InfrastructureFeatureEngine,
    terrain_engine: TerrainFeatureEngine,
    constraint_engine: LandConstraintEngine,
    prospectivity_score=None,
):
    """
    Computes the full Mining Efficiency Score for one point.

    prospectivity_score: pass a real value from Layer 2 once available.
    If None, falls back to the mock generator so this function is
    fully runnable today.
    """
    if prospectivity_score is None:
        prospectivity_score = get_mock_prospectivity_score(lon, lat)

    infra = infra_engine.get_infrastructure_features(lon, lat)
    terrain = terrain_engine.get_terrain_features(lon, lat)
    constraints = constraint_engine.get_constraint_features(lon, lat)

    # --- Sub-scores, each normalized to 0-1 ---
    accessibility_score = np.mean([
        normalize_distance(infra["distance_to_road_m"], max_acceptable_m=20000),
        normalize_distance(infra["distance_to_railway_m"], max_acceptable_m=50000),
    ])

    terrain_score = normalize_slope(terrain["slope_deg"])

    infrastructure_score = np.mean([
        normalize_distance(infra["distance_to_power_m"], max_acceptable_m=15000),
        normalize_distance(infra["distance_to_existing_mine_m"], max_acceptable_m=30000),
        normalize_distance(infra["distance_to_processing_facility_m"], max_acceptable_m=40000),
    ])

    land_eligibility_score = float(constraints["land_eligible"])

    # --- Weighted combination ---
    efficiency = (
        EFFICIENCY_WEIGHTS["prospectivity"] * prospectivity_score
        + EFFICIENCY_WEIGHTS["accessibility"] * accessibility_score
        + EFFICIENCY_WEIGHTS["terrain"] * terrain_score
        + EFFICIENCY_WEIGHTS["infrastructure"] * infrastructure_score
        + EFFICIENCY_WEIGHTS["land_eligibility"] * land_eligibility_score
    )

    # Hard constraint override: a restricted area should never rank high,
    # regardless of how good the other scores are.
    if constraints["is_restricted_area"]:
        efficiency *= 0.1  # heavily penalize rather than silently drop —
                            # keeps it visible in the ranking as "flagged"

    return {
        "longitude": lon,
        "latitude": lat,
        "prospectivity_score": round(prospectivity_score, 3),
        "accessibility_score": round(accessibility_score, 3),
        "terrain_score": round(terrain_score, 3),
        "infrastructure_score": round(infrastructure_score, 3),
        "land_eligibility_score": round(land_eligibility_score, 3),
        "is_restricted_area": bool(constraints["is_restricted_area"]),
        "mining_efficiency_score": round(efficiency * 100, 1),  # 0-100 scale
    }


if __name__ == "__main__":
    # Fully runnable today with mock data, no Layer 2 needed
    infra_engine = InfrastructureFeatureEngine(
        roads_path="data/roads.geojson",
        railways_path="data/railways.geojson",
        power_lines_path="data/power_lines.geojson",
        existing_mines_csv="data/existing_mines.csv",
        processing_facilities_csv="data/processing_facilities.csv",
    )
    terrain_engine = TerrainFeatureEngine(dem_path="data/processed/dem.tif")
    constraint_engine = LandConstraintEngine(protected_areas_path="data/protected_areas.geojson")

    result = compute_mining_efficiency(
        lon=83.5, lat=21.2,
        infra_engine=infra_engine,
        terrain_engine=terrain_engine,
        constraint_engine=constraint_engine,
        # prospectivity_score left out on purpose -> uses mock value for now
    )
    print(result)
