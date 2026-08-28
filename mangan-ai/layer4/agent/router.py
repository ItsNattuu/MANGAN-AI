def route_request(request):

    request = request.lower().strip()

    if (
        "top 5" in request
        or "top five" in request
        or "top targets" in request
    ):
        return "TOP_TARGETS"

    if (
        "why" in request
        or "explain" in request
    ):
        return "EXPLAIN"

    if (
        "analyze" in request
        or "analyse" in request
    ):
        return "ANALYZE"

    if (
        "best" in request
        or "highest" in request
        or "priority" in request
    ):
        return "BEST_TARGET"

    return "BEST_TARGET"
