"""
Central configuration for Layer 2.

Change the study area here, not in individual scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RASTERS = ROOT / "rasters"
OUTPUTS = ROOT / "outputs"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, RASTERS, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ study area
# Balaghat / Sausar belt, MP-Maharashtra
AOI_BOUNDS = (79.7, 21.0, 80.9, 22.3)  # lon_min, lat_min, lon_max, lat_max
CRS = "EPSG:32644"  # UTM 44N — correct for 79.7-80.9E
GRID_SIZE_M = 100

# ------------------------------------------------------------ validation
# Spatial block size for cross-validation. ~0.09 deg =~ 10km at this latitude.
BLOCK_SIZE_DEG = 0.09
N_FOLDS = 5
RANDOM_SEED = 42

# PU learning
PU_N_ESTIMATORS = 100  # number of bagging rounds
N_UNLABELED_SAMPLE = 20000  # unlabeled points to sample for training

# ------------------------------------------------------- feature groups
FEATURE_GROUPS = {
    "A_remote_sensing": [
        "B02", "B03", "B04", "B08", "B11", "B12",
        "Red_NIR_Ratio", "SWIR_Ratio", "Visible_SWIR_Ratio",
        "Red_SWIR_Index", "Mn_Score",
    ],
    "B_geology": [
        "lithology", "geological_age",
    ],
    "C_structure": [
        "distance_to_fault", "distance_to_lineament",
        "fault_density", "distance_to_contact",
    ],
    "D_terrain": [
        "elevation", "slope", "aspect_sin", "aspect_cos",
        "terrain_ruggedness",
    ],
    "E_geochemistry": [
        # ONLY populate if real measured data is obtained.
        # Leave empty rather than synthesising values.
    ],
    "F_occurrence_context": [
        "Mn_occurrence_density_5km", "Mn_occurrence_density_10km",
    ],
}

CATEGORICAL_FEATURES = ["lithology", "geological_age"]

# ------------------------------------------------------- CONTEXT COLUMNS
# Passed THROUGH to the output CSV for Layer 3 (mining feasibility) to
# consume. They are NOT model features and must never be trained on.
#
# distance_to_mine in particular is pure leakage as a feature — mines
# exist precisely where manganese was already found. Keeping these in a
# separate list is what stops them silently entering the feature matrix.
CONTEXT_COLUMNS = [
    "distance_to_road",
    "distance_to_rail",
    "distance_to_mine",
    "distance_to_power",
    "distance_to_processing",
    "land_use",
    "protected_area",
]

# Derived summary column reported in the output. Excluded from training
# because it is computed FROM other features (lithology, fault distance,
# Mn_Score) — including it would be redundant and collinear.
OUTPUT_ONLY_COLUMNS = ["geological_favorability"]

# ---------------------------------------------------------------- BANNED
# These predict the label directly or encode human activity rather than
# geology. Never add them to FEATURE_GROUPS.
BANNED_FEATURES = [
    "distance_to_known_Mn_occurrence",
    "distance_to_nearest_deposit",
    "distance_to_mine",
    "roads", "railways", "powerlines",
    "night_lights", "built_up_index",
    "latitude", "longitude",  # raw coords -> memorisation
    "mining_footprint",
]


def all_features(include_geochem: bool = False) -> list:
    """Flatten feature groups into a single list."""
    feats = []
    for group, cols in FEATURE_GROUPS.items():
        if group == "E_geochemistry" and not include_geochem:
            continue
        feats.extend(cols)
    return feats


def assert_no_banned(columns) -> None:
    """Raise if any banned feature snuck into the table."""
    found = [c for c in columns if c in BANNED_FEATURES]
    if found:
        raise ValueError(
            f"BANNED features present in feature matrix: {found}\n"
            "These leak the label. Remove them before training."
        )
