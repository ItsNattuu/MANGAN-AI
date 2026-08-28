def choose_best_target(targets):

    if not targets:
        return None

    # Do not recommend restricted targets.
    eligible = [
        target
        for target in targets
        if not target.get(
            "is_restricted_area",
            False
        )
    ]

    if not eligible:
        return None

    # Layer 3 has already calculated
    # overall_priority.
    eligible.sort(
        key=lambda target: float(
            target["overall_priority"]
        ),
        reverse=True
    )

    return eligible[0]


def get_top_targets(
    targets,
    n=5
):

    eligible = [
        target
        for target in targets
        if not target.get(
            "is_restricted_area",
            False
        )
    ]

    eligible.sort(
        key=lambda target: float(
            target["overall_priority"]
        ),
        reverse=True
    )

    return eligible[:n]


def explain_decision(target):

    if target is None:
        return []

    reasons = []

    prospectivity = float(
        target.get(
            "prospectivity_score",
            0
        )
    )

    # Handle either 0-1 or 0-100 format.
    if prospectivity <= 1:
        prospectivity *= 100

    efficiency = float(
        target.get(
            "mining_efficiency_score",
            0
        )
    )

    priority = float(
        target.get(
            "overall_priority",
            0
        )
    )

    reasons.append(
        f"Manganese prospectivity is "
        f"{prospectivity:.1f}/100."
    )

    reasons.append(
        f"Mining efficiency is "
        f"{efficiency:.1f}/100."
    )

    reasons.append(
        f"Overall priority is "
        f"{priority:.1f}/100."
    )

    if target.get(
        "is_restricted_area",
        False
    ):
        reasons.append(
            "The target is inside a "
            "restricted area."
        )
    else:
        reasons.append(
            "The target is not flagged "
            "as a restricted area."
        )

    return reasons


def make_recommendation(target):

    if target is None:
        return (
            "No eligible target could be "
            "recommended from the supplied data."
        )

    return (
        "Prioritize this target for "
        "geological field validation before "
        "any drilling or mining decision."
    )
