from pathlib import Path

import pandas as pd

from .raw_2_excel import read_raw_image
from .manganese_features import calculate_manganese_features


def identify_manganese(
    image_path: str,
    output_excel: str
) -> dict:
    """
    Complete Layer 1 manganese identification pipeline.

    Workflow
    --------
    Raw multispectral image
            ↓
    Band extraction
            ↓
    Spectral feature calculation
            ↓
    Prototype manganese score
            ↓
    Manganese classification
            ↓
    Excel output

    Parameters
    ----------
    image_path : str
        Path to the multispectral raster.

    output_excel : str
        Path where the processed Excel file will be written.

    Returns
    -------
    dict
        Summary of the Layer 1 processing result.
    """

    # -------------------------------------------------
    # 1. Read raw satellite image
    # -------------------------------------------------

    print("Reading satellite image...")

    df = read_raw_image(image_path)

    print(f"Read {len(df)} pixels.")

    # -------------------------------------------------
    # 2. Calculate manganese spectral features
    # -------------------------------------------------

    print("Calculating manganese features...")

    df = calculate_manganese_features(df)

    # -------------------------------------------------
    # 3. Save processed Layer 1 dataset
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

    print(
        f"Layer 1 output saved to: {output_path}"
    )

    # -------------------------------------------------
    # 4. Generate summary statistics
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

    # -------------------------------------------------
    # 5. Return structured result
    # -------------------------------------------------

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
