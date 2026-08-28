import pandas as pd

from .mining_efficiency import (
    compute_mining_efficiency
)

from .infrastructure_features import (
    InfrastructureFeatureEngine
)

from .land_constraints import (
    LandConstraintEngine
)


def run_layer3_from_layer2(
    layer2_df
):

    rows = []

    # -----------------------------------------
    # Infrastructure data
    # -----------------------------------------

    infra = InfrastructureFeatureEngine(

        roads_path="layer3/data/roads.geojson",

        railways_path="layer3/data/railways.geojson",

        power_lines_path="layer3/data/power_lines.geojson",

        existing_mines_csv=
            "layer3/data/existing_mines.csv",

        processing_facilities_csv=
            "layer3/data/processing_facilities.csv"
    )

    # -----------------------------------------
    # Constraints
    # -----------------------------------------

    constraints = LandConstraintEngine(

        protected_areas_path=
            "layer3/data/protected_areas.geojson"
    )

    # -----------------------------------------
    # IMPORTANT
    # -----------------------------------------

    # Your repository currently references
    # TerrainFeatureEngine in mining_efficiency.py.
    #
    # Make sure terrain_features.py exists
    # and is importable before running this.

    from .terrain_features import (
        TerrainFeatureEngine
    )

    terrain = TerrainFeatureEngine()

    # -----------------------------------------
    # Process every Layer 2 target
    # -----------------------------------------

    for _, row in layer2_df.iterrows():

        result = compute_mining_efficiency(

            lon=float(row["longitude"]),

            lat=float(row["latitude"]),

            infra_engine=infra,

            terrain_engine=terrain,

            constraint_engine=constraints,

            prospectivity_score=
                float(row["Mn_Probability"]),
            elevation=row.get("elevation"),
            slope=row.get("slope")
        )

        # Add Layer 2 information
        result["Mn_Probability"] = (
            float(row["Mn_Probability"])
        )

        result["Mn_Prospectivity"] = (
            int(row["Mn_Prospectivity"])
        )

        result[
            "geological_favorability"
        ] = float(
            row[
                "geological_favorability"
            ]
        )

        rows.append(result)

    result_df = pd.DataFrame(
        rows
    )

    # -----------------------------------------
    # Overall priority
    # -----------------------------------------

    result_df[
        "overall_priority"
    ] = (

        0.5 *
        result_df[
            "Mn_Prospectivity"
        ]

        +

        0.5 *
        result_df[
            "mining_efficiency_score"
        ]
    )

    result_df[
        "overall_priority"
    ] = result_df[
        "overall_priority"
    ].round(1)

    return result_df.sort_values(
        "overall_priority",
        ascending=False
    ).reset_index(drop=True)
