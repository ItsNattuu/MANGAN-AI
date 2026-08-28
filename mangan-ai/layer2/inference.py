import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from . import config
from .pu_learning import BaggingPU

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def run_prediction_from_dataframe(
    df: pd.DataFrame
):

    if XGBClassifier is None:
        raise ImportError(
            "xgboost is required. "
            "Install it using: pip install xgboost"
        )

    df = df.copy()

    feature_columns = config.all_features()

    missing = [
        c for c in feature_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Layer 1 is missing Layer 2 features: {missing}"
        )

    X = df[feature_columns].astype(float)

    X = X.fillna(
        X.mean()
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # ------------------------------------------------
    # DEMO MODEL
    # ------------------------------------------------
    #
    # This is only a pipeline test.
    # Replace with the trained Layer 2 model.
    #

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )

    # For now generate a deterministic
    # pseudo-label from Mn_Score so the
    # pipeline can be tested.
    y = (
        df["Mn_Score"] >=
        df["Mn_Score"].median()
    ).astype(int)

    model.fit(
        X_scaled,
        y
    )

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    output = df[
        ["Latitude", "Longitude"]
    ].copy()

    output = output.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude"
        }
    )

    output["Mn_Probability"] = (
        probabilities.round(4)
    )

    output["Mn_Prospectivity"] = (
        probabilities * 100
    ).round(0).astype(int)

    output["geological_favorability"] = (
        df["Mn_Score"] / 100
    ).round(3)

    return output
