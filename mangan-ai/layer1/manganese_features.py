import numpy as np
import pandas as pd

def safe_ratio(a, b):
"""
Calculate a ratio while preventing division by zero.
"""

```
a = np.asarray(a, dtype=float)
b = np.asarray(b, dtype=float)

return np.divide(
    a,
    b,
    out=np.zeros_like(a, dtype=float),
    where=b != 0
)
```

def calculate_manganese_features(
df: pd.DataFrame
) -> pd.DataFrame:
"""
Calculate prototype manganese spectral features.

```
Expected Sentinel-style bands:

    B02 - Blue
    B03 - Green
    B04 - Red
    B08 - NIR
    B11 - SWIR 1
    B12 - SWIR 2

These features are intended for the prototype and
should not be presented as laboratory-grade manganese
detection.

Returns
-------
pandas.DataFrame
    Original dataframe plus manganese spectral
    features, Mn_Score and Mn_Class.
"""

# ---------------------------------------------------------
# 1. Required spectral bands
# ---------------------------------------------------------

required = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "B12"
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing required spectral bands: "
        f"{missing}"
    )

# ---------------------------------------------------------
# 2. Convert bands to numeric arrays
# ---------------------------------------------------------

B02 = pd.to_numeric(
    df["B02"],
    errors="coerce"
).fillna(0).to_numpy(dtype=float)

B03 = pd.to_numeric(
    df["B03"],
    errors="coerce"
).fillna(0).to_numpy(dtype=float)

B04 = pd.to_numeric(
    df["B04"],
    errors="coerce"
).fillna(0).to_numpy(dtype=float)

B08 = pd.to_numeric(
    df["B08"],
    errors="coerce"
).fillna(0).to_numpy(dtype=float)

B11 = pd.to_numeric(
    df["B11"],
    errors="coerce"
).fillna(0).to_numpy(dtype=float)

B12 = pd.to_numeric(
    df["B12"],
    errors="coerce"
).fillna(0).to_numpy(dtype=float)

# ---------------------------------------------------------
# 3. Spectral features
# ---------------------------------------------------------

# Red / NIR relationship
df["Red_NIR_Ratio"] = safe_ratio(
    B04,
    B08
)

# SWIR relationship
df["SWIR_Ratio"] = safe_ratio(
    B11,
    B12
)

# Visible / SWIR relationship
df["Visible_SWIR_Ratio"] = safe_ratio(
    B04 + B03,
    B11 + B12
)

# Red / SWIR normalized difference
df["Red_SWIR_Index"] = safe_ratio(
    B04 - B11,
    B04 + B11
)

# ---------------------------------------------------------
# 4. Prototype manganese score
# ---------------------------------------------------------

raw_score = (
    0.30 * df["Red_NIR_Ratio"]
    \+ 0.30 * df["SWIR_Ratio"]
    \+ 0.20 * df["Visible_SWIR_Ratio"]
    \+ 0.20 * df["Red_SWIR_Index"]
)

raw_score = pd.Series(
    raw_score,
    index=df.index
)

# ---------------------------------------------------------
# 5. Normalize score to 0-100
# ---------------------------------------------------------

minimum = raw_score.min()
maximum = raw_score.max()

if (
    not np.isfinite(minimum)
    or not np.isfinite(maximum)
    or maximum == minimum
):
    df["Mn_Score"] = 0.0

else:
    df["Mn_Score"] = (
        (raw_score - minimum)
        / (maximum - minimum)
    ) * 100.0

# ---------------------------------------------------------
# 6. Classification
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# 7. Make score numeric
# ---------------------------------------------------------

df["Mn_Score"] = pd.to_numeric(
    df["Mn_Score"],
    errors="coerce"
).fillna(0.0)

return df
```
