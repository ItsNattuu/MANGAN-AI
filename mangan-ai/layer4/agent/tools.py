from .decision import (
    choose_best_target,
    get_top_targets,
    explain_decision,
)


def find_best_target(
    layer3_results
):

    return choose_best_target(
        layer3_results
    )


def find_top_targets(
    layer3_results,
    n=5
):

    return get_top_targets(
        layer3_results,
        n
    )


def explain_target(
    target
):

    return explain_decision(
        target
    )
