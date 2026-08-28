import numpy as np
import pandas as pd


def safe_ratio(a, b):
    """
    Calculate a ratio while preventing division by zero.
    """
    return np.divide(
        a,
        b,
        out=np.zeros_like(a, dtype=float),
        where=b != 0
    )


def calculate_manganese_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    required = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required bands: {missing}"
        )

    B2 = df["B2"].astype(float).values
    B3 = df["B3"].astype(float).values
    B4 = df["B4"].astype(float).values
    B8 = df["B8"].astype(float).values
    B11 = df["B11"].astype(float).values
    B12 = df["B12"].astype(float).values

    # -------------------------------------------------
    # Spectral features
    # -------------------------------------------------

    df["Red_NIR_Ratio"] = safe_ratio(
        B4,
        B8
    )

    df["SWIR_Ratio"] = safe_ratio(
        B11,
        B12
    )

    df["Visible_SWIR_Ratio"] = safe_ratio(
        B4 + B3,
        B11 + B12
    )

    # Normalized difference between red and SWIR
    df["Red_SWIR_Index"] = (
        safe_ratio(
            B4 - B11,
            B4 + B11
        )
    )

    # -------------------------------------------------
    # Prototype manganese spectral score
    # -------------------------------------------------

    raw_score = (
        0.30 * df["Red_NIR_Ratio"] +
        0.30 * df["SWIR_Ratio"] +
        0.20 * df["Visible_SWIR_Ratio"] +
        0.20 * df["Red_SWIR_Index"]
    )

    # Normalize to 0–100
    minimum = raw_score.min()
    maximum = raw_score.max()

    if maximum == minimum:

        df["Mn_Score"] = 0

    else:

        df["Mn_Score"] = (
            (raw_score - minimum) /
            (maximum - minimum)
        ) * 100

    # -------------------------------------------------
    # Classification
    # -------------------------------------------------

    df["Mn_Class"] = pd.cut(
        df["Mn_Score"],
        bins=[
            -np.inf,
            30,
            60,
            np.inf
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH"
        ]
    )

    return df
