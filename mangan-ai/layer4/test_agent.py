from agent.agent import analyze


# ============================================================
# MOCK LAYER 2 OUTPUT
# ============================================================

layer2_results = [

    {
        "latitude": 21.1500,
        "longitude": 83.4200,
        "Mn_Probability": 0.91,
        "Mn_Prospectivity": 0.91,
    },

    {
        "latitude": 21.3000,
        "longitude": 83.5500,
        "Mn_Probability": 0.86,
        "Mn_Prospectivity": 0.86,
    },

    {
        "latitude": 20.9500,
        "longitude": 83.6100,
        "Mn_Probability": 0.79,
        "Mn_Prospectivity": 0.79,
    }

]


# ============================================================
# MOCK LAYER 3 OUTPUT
# ============================================================

layer3_results = [

    {
        "latitude": 21.1500,
        "longitude": 83.4200,

        "prospectivity_score": 0.91,

        "mining_efficiency_score": 78.0,

        "overall_priority": 84.5,

        "is_restricted_area": False,
    },

    {
        "latitude": 21.3000,
        "longitude": 83.5500,

        "prospectivity_score": 0.86,

        "mining_efficiency_score": 91.0,

        "overall_priority": 88.5,

        "is_restricted_area": False,
    },

    {
        "latitude": 20.9500,
        "longitude": 83.6100,

        "prospectivity_score": 0.79,

        "mining_efficiency_score": 82.0,

        "overall_priority": 80.5,

        "is_restricted_area": False,
    }

]


# ============================================================
# RUN LAYER 4
# ============================================================

result = analyze(

    request=(
        "Find the best manganese "
        "exploration target"
    ),

    layer2_results=layer2_results,

    layer3_results=layer3_results
)


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 55)
print("              MANGAN-AI LAYER 4")
print("=" * 55)


print(
    "\nStatus:",
    result["status"]
)

print(
    "Decision:",
    result["decision"]
)

print(
    "Intent:",
    result["intent"]
)


# ============================================================
# BEST TARGET
# ============================================================

best = result["best_target"]


if best:

    print()
    print("-" * 55)
    print("                    BEST TARGET")
    print("-" * 55)

    print(
        "Latitude:",
        best["latitude"]
    )

    print(
        "Longitude:",
        best["longitude"]
    )

    prospectivity = (
        best["prospectivity_score"]
    )

    if prospectivity <= 1:
        prospectivity *= 100

    print(
        "Manganese Prospectivity:",
        f"{prospectivity:.1f}/100"
    )

    print(
        "Mining Efficiency:",
        best["mining_efficiency_score"]
    )

    print(
        "Overall Priority:",
        best["overall_priority"]
    )

    print(
        "Restricted:",
        best["is_restricted_area"]
    )


# ============================================================
# REASONING
# ============================================================

print()
print("-" * 55)
print("                    WHY?")
print("-" * 55)


for reason in result["reasoning"]:

    print(
        "•",
        reason
    )


# ============================================================
# RECOMMENDATION
# ============================================================

print()
print("-" * 55)
print("                 RECOMMENDATION")
print("-" * 55)

print(
    result["recommendation"]
)


# ============================================================
# TOP 5
# ============================================================

print()
print("-" * 55)
print("                   TOP TARGETS")
print("-" * 55)


for index, target in enumerate(
    result["top_targets"],
    start=1
):

    print(
        f"{index}. "
        f"({target['latitude']}, "
        f"{target['longitude']}) "
        f"→ Priority: "
        f"{target['overall_priority']}"
    )


print()
print("=" * 55)
print("                  TEST COMPLETE")
print("=" * 55)
