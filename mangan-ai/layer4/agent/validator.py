REQUIRED_LAYER3_FIELDS = [
    "latitude",
    "longitude",
    "prospectivity_score",
    "mining_efficiency_score",
    "overall_priority",
    "is_restricted_area",
]


def validate_layer3_target(target):

    missing = [
        field
        for field in REQUIRED_LAYER3_FIELDS
        if field not in target
    ]

    if missing:
        return False, missing

    return True, []


def validate_targets(layer3_results):

    valid = []
    rejected = []

    for target in layer3_results:

        valid_target, missing = (
            validate_layer3_target(target)
        )

        if valid_target:
            valid.append(target)

        else:
            rejected.append({
                "target": target,
                "missing": missing
            })

    return valid, rejected
