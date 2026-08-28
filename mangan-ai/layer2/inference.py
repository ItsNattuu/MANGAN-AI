import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_inference_features(
    df: pd.DataFrame,
    feature_columns,
):
    """
    Prepare Layer 2 features for prototype inference.

    The preprocessing is intentionally kept simple because
    this version is being used for the synthetic-data demo.

    Steps:
        1. Select configured Layer 2 features
        2. Convert values to numeric
        3. Replace inf values
        4. Median imputation
        5. Standardization
    """

    X = df[
        feature_columns
    ].copy()

    # -----------------------------------------------------
    # Convert all features to numeric
    # -----------------------------------------------------

    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Replace invalid infinite values
    # -----------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # Median imputation
    # -----------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = imputer.fit_transform(
        X
    )

    # -----------------------------------------------------
    # Standardization
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_imputed
    )

    return X_scaled


# =========================================================
# PROTOTYPE MODEL
# =========================================================

def build_demo_model():
    """
    Create the deterministic XGBoost model used by the
    synthetic-data demonstration.

    This is NOT the final trained production model.
    """

    if XGBClassifier is None:

        raise ImportError(
            "xgboost is required for the Layer 2 "
            "prototype. Install it using:\n\n"
            "pip install xgboost"
        )

    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss"
    )


# =========================================================
# MAIN INFERENCE FUNCTION
# =========================================================

def run_prediction_from_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate Layer 2 manganese prospectivity predictions
    from a Layer 1 dataframe.

    Prototype workflow:

        Layer 1 dataframe
                ↓
        Layer 2 features
                ↓
        preprocessing
                ↓
        synthetic labels
                ↓
        XGBoost
                ↓
        prospectivity probability
                ↓
        Layer 3-compatible dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        Layer 1 processed dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing:

            latitude
            longitude
            Mn_Probability
            Mn_Prospectivity
            geological_favorability
    """

    # -----------------------------------------------------
    # 1. Validate input
    # -----------------------------------------------------

    if df is None:

        raise ValueError(
            "Layer 2 received an empty dataframe."
        )

    if df.empty:

        raise ValueError(
            "Layer 2 received a dataframe with zero rows."
        )

    df = df.copy()

    # -----------------------------------------------------
    # 2. Required geographic columns
    # -----------------------------------------------------

    coordinate_columns = [
        "Latitude",
        "Longitude",
    ]

    missing_coordinates = [
        column
        for column in coordinate_columns
        if column not in df.columns
    ]

    if missing_coordinates:

        raise ValueError(
            "Layer 1 output is missing geographic "
            f"columns: {missing_coordinates}"
        )

    # -----------------------------------------------------
    # 3. Get Layer 2 feature schema
    # -----------------------------------------------------

    configured_features = config.all_features()

    # -----------------------------------------------------
    # 4. Protect against accidental Layer 3 leakage
    # -----------------------------------------------------

    config.assert_no_banned(
        configured_features
    )

    # -----------------------------------------------------
    # 5. Check required features
    # -----------------------------------------------------

    missing = [
        column
        for column in configured_features
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Layer 1 is missing Layer 2 features:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # -----------------------------------------------------
    # 6. Prepare features
    # -----------------------------------------------------

    X_scaled = prepare_inference_features(
        df,
        configured_features,
    )

    # -----------------------------------------------------
    # 7. Create deterministic prototype labels
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # We are using fake data for the demonstration.
    #
    # Mn_Score is therefore used as a temporary proxy
    # to generate pseudo-labels.
    #
    # This MUST NOT be described as real geological
    # ground truth.
    # -----------------------------------------------------

    if "Mn_Score" not in df.columns:

        raise ValueError(
            "Mn_Score is required for the current "
            "synthetic-data prototype."
        )

    mn_score = pd.to_numeric(
        df["Mn_Score"],
        errors="coerce"
    ).fillna(0.0)

    threshold = mn_score.median()

    y = (
        mn_score >= threshold
    ).astype(int)

    # -----------------------------------------------------
    # 8. Safety check
    # -----------------------------------------------------

    if y.nunique() < 2:

        raise ValueError(
            "Synthetic Layer 2 labels contain only one "
            "class. Mn_Score must contain enough variation "
            "for the prototype model to train."
        )

    # -----------------------------------------------------
    # 9. Build demo model
    # -----------------------------------------------------

    model = build_demo_model()

    # -----------------------------------------------------
    # 10. Train prototype model
    # -----------------------------------------------------

    model.fit(
        X_scaled,
        y
    )

    # -----------------------------------------------------
    # 11. Generate probability
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    # -----------------------------------------------------
    # 12. Construct Layer 3 input
    # -----------------------------------------------------

    output = pd.DataFrame(
        {
            "latitude": pd.to_numeric(
                df["Latitude"],
                errors="coerce"
            ),

            "longitude": pd.to_numeric(
                df["Longitude"],
                errors="coerce"
            ),

            "Mn_Probability": probabilities,

            "Mn_Prospectivity": (
                probabilities * 100
            ).round(0).astype(int),

            "geological_favorability": (
                mn_score / 100
            ).round(3),
        }
    )

    # -----------------------------------------------------
    # 13. Preserve useful identifiers
    # -----------------------------------------------------

    if "Pixel_X" in df.columns:

        output["Pixel_X"] = df[
            "Pixel_X"
        ].values

    if "Pixel_Y" in df.columns:

        output["Pixel_Y"] = df[
            "Pixel_Y"
        ].values

    if "target_id" in df.columns:

        output["target_id"] = df[
            "target_id"
        ].values

    # -----------------------------------------------------
    # 14. Remove invalid coordinates
    # -----------------------------------------------------

    output = output.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    ).reset_index(
        drop=True
    )

    return output
