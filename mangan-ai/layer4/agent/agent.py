from .state import AgentState

from .router import route_request

from .validator import validate_targets

from .tools import (
    find_best_target,
    find_top_targets,
    explain_target,
)

from .decision import (
    make_recommendation,
)


def analyze(
    request,
    layer2_results,
    layer3_results
):

    # --------------------------------------------------------
    # CREATE AGENT STATE
    # --------------------------------------------------------

    state = AgentState(

        request=request,

        layer2_results=layer2_results,

        layer3_results=layer3_results
    )

    # --------------------------------------------------------
    # 1. UNDERSTAND USER REQUEST
    # --------------------------------------------------------

    intent = route_request(
        request
    )

    # --------------------------------------------------------
    # 2. VALIDATE LAYER 3 RESULTS
    # --------------------------------------------------------

    (
        valid_targets,
        rejected_targets
    ) = validate_targets(
        layer3_results
    )

    state.validated_targets = (
        valid_targets
    )

    # --------------------------------------------------------
    # 3. SELECT TARGETS
    # --------------------------------------------------------

    if intent == "TOP_TARGETS":

        selected_targets = (
            find_top_targets(
                valid_targets,
                n=5
            )
        )

        best = (
            selected_targets[0]
            if selected_targets
            else None
        )

    else:

        best = find_best_target(
            valid_targets
        )

        selected_targets = (
            find_top_targets(
                valid_targets,
                n=5
            )
        )

    state.best_target = best

    # --------------------------------------------------------
    # 4. EXPLAIN DECISION
    # --------------------------------------------------------

    if best:

        state.reasoning = (
            explain_target(
                best
            )
        )

        state.decision = (
            "PRIORITIZE"
        )

        state.recommendation = (
            make_recommendation(
                best
            )
        )

    else:

        state.decision = (
            "NO_ELIGIBLE_TARGET"
        )

        state.recommendation = (
            "No eligible target could be "
            "recommended from the supplied data."
        )

    # --------------------------------------------------------
    # 5. RETURN STRUCTURED RESULT
    # --------------------------------------------------------

    return {

        "status": "success",

        "intent": intent,

        "decision": state.decision,

        "best_target": state.best_target,

        "top_targets": selected_targets,

        "reasoning": state.reasoning,

        "recommendation": (
            state.recommendation
        ),

        "rejected_targets": (
            rejected_targets
        )

    }
