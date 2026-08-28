"""
ranking.py
------------
Layer 3, Part D: turn a set of candidate points into a ranked target
list — the final output format described in the project plan
(coordinates + prospectivity + efficiency + overall priority).

Runnable right now with a synthetic list of sample points, so you
don't need Layer 2's clustering output to test this logic. Once
Layer 2 produces real target-zone centroids, just pass those in
instead of the synthetic list.
"""

import pandas as pd

from infrastructure_features import InfrastructureFeatureEngine
from land_constraints import LandConstraintEngine
from terrain_features import TerrainFeatureEngine
from mining_efficiency import compute_mining_efficiency

OVERALL_PRIORITY_WEIGHTS = {
    "prospectivity": 0.5,
    "mining_efficiency": 0.5,
}


def rank_targets(candidate_points, infra_engine, terrain_engine, constraint_engine, prospectivity_lookup=None):
    """
    candidate_points: list of (lon, lat) tuples. In the final system
      these come from Layer 2's clustering step (target zone centroids).
      For now, you can test with any list of points in your study area.

    prospectivity_lookup: optional dict {(lon, lat): score}. If not
      provided, mining_efficiency.py falls back to its mock generator.
    """
    rows = []
    for lon, lat in candidate_points:
        prospectivity_score = None
        if prospectivity_lookup is not None:
            prospectivity_score = prospectivity_lookup.get((lon, lat))

        result = compute_mining_efficiency(
            lon, lat,
            infra_engine=infra_engine,
            terrain_engine=terrain_engine,
            constraint_engine=constraint_engine,
            prospectivity_score=prospectivity_score,
        )

        overall_priority = (
            OVERALL_PRIORITY_WEIGHTS["prospectivity"] * result["prospectivity_score"] * 100
            + OVERALL_PRIORITY_WEIGHTS["mining_efficiency"] * result["mining_efficiency_score"]
        ) 

        result["overall_priority"] = round(overall_priority, 1)
        rows.append(result)

    df = pd.DataFrame(rows).sort_values("overall_priority", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def print_top_targets(df, top_n=5):
    for _, row in df.head(top_n).iterrows():
        flag = " [RESTRICTED AREA]" if row["is_restricted_area"] else ""
        print(
            f"Target #{int(row['rank'])}{flag}\n"
            f"  Coordinates: {row['latitude']:.4f}, {row['longitude']:.4f}\n"
            f"  Prospectivity: {row['prospectivity_score']*100:.0f}/100\n"
            f"  Mining efficiency: {row['mining_efficiency_score']:.0f}/100\n"
            f"  Overall priority: {row['overall_priority']:.0f}/100\n"
        )


if __name__ == "__main__":
    # SYNTHETIC test points — replace with Layer 2's cluster centroids later
    synthetic_candidates = [
        (83.42, 21.15),
        (83.55, 21.30),
        (83.61, 20.95),
        (84.02, 21.40),
        (83.88, 21.05),
    ]

    infra_engine = InfrastructureFeatureEngine(
        roads_path="data/roads.geojson",
        railways_path="data/railways.geojson",
        power_lines_path="data/power_lines.geojson",
        existing_mines_csv="data/existing_mines.csv",
        processing_facilities_csv="data/processing_facilities.csv",
    )
    terrain_engine = TerrainFeatureEngine(dem_path="data/processed/dem.tif")
    constraint_engine = LandConstraintEngine(protected_areas_path="data/constraints/protected_areas.geojson")

    ranked = rank_targets(synthetic_candidates, infra_engine, terrain_engine, constraint_engine)
    print_top_targets(ranked)
    ranked.to_csv("outputs/predictions/ranked_targets.csv", index=False)
