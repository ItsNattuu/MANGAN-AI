from pathlib import Path

import pandas as pd

from raw_2excel import read_raw_image
from manganese_features import calculate_manganese_features


def identify_manganese(
    image_path: str,
    output_excel: str
) -> dict:
    """
    Complete manganese identification pipeline.

    Raw image
        ↓
    Band extraction
        ↓
    Spectral feature calculation
        ↓
    Manganese score
        ↓
    Excel output
    """

    print("Reading satellite image...")

    df = read_raw_image(image_path)

    print(
        f"Read {len(df)} pixels."
    )

    print("Calculating manganese features...")

    df = calculate_manganese_features(df)

    # -------------------------------------------------
    # Save result
    # -------------------------------------------------

    output_path = Path(output_excel)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_excel(
        output_path,
        index=False
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    high = int(
        (df["Mn_Class"] == "HIGH").sum()
    )

    medium = int(
        (df["Mn_Class"] == "MEDIUM").sum()
    )

    low = int(
        (df["Mn_Class"] == "LOW").sum()
    )

    highest_score = float(
        df["Mn_Score"].max()
    )

    return {
        "status": "success",
        "image": str(image_path),
        "output": str(output_path),
        "pixels_processed": len(df),
        "high_prospectivity_pixels": high,
        "medium_prospectivity_pixels": medium,
        "low_prospectivity_pixels": low,
        "highest_mn_score": round(
            highest_score,
            2
        )
    }
