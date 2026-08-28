def calculate_priority(layer2, layer3):
    """
    Calculate a combined priority score.

    This is a prototype scoring method.
    It should later be replaced or calibrated
    using validated domain knowledge / ML outputs.
    """

    manganese_score = float(
        layer2["Mn_Prospectivity"]
    )

    mining_score = float(
        layer3["Overall Priority"]
    )

    # 60% geological/manganese potential
    # 40% practical mining priority

    final_score = (
        0.60 * manganese_score
        +
        0.40 * mining_score
    )

    return round(
        final_score,
        2
    )


def rank_targets(layer2_df, layer3_df):

    ranked_targets = []

    for _, layer2 in layer2_df.iterrows():

        target_id = int(
            layer2.name + 1
        )

        layer3_rows = layer3_df[
            layer3_df["Target"] == target_id
        ]

        if layer3_rows.empty:
            continue

        layer3 = layer3_rows.iloc[0]

        score = calculate_priority(
            layer2,
            layer3
        )

        ranked_targets.append({

            "target": target_id,

            "latitude":
                layer2["latitude"],

            "longitude":
                layer2["longitude"],

            "manganese_probability":
                layer2["Mn_Probability"],

            "manganese_prospectivity":
                layer2["Mn_Prospectivity"],

            "geological_favorability":
                layer2[
                    "geological_favorability"
                ],

            "overall_mining_priority":
                layer3[
                    "Overall Priority"
                ],

            "final_priority_score":
                score,

            "status":
                layer3["Status"]
        })

    ranked_targets.sort(
        key=lambda x:
            x["final_priority_score"],
        reverse=True
    )

    return ranked_targets
