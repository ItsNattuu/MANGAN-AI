from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def read_raw_image(image_path: str) -> pd.DataFrame:
    """
    Read a multispectral raster and convert its pixels
    into a Pandas DataFrame.

    Each row represents one pixel.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with rasterio.open(image_path) as src:

        image = src.read()

        width = src.width
        height = src.height
        band_count = src.count

        # Pixel coordinates
        x, y = np.meshgrid(
            np.arange(width),
            np.arange(height)
        )

        data = {
            "Pixel_X": x.flatten(),
            "Pixel_Y": y.flatten(),
        }

        # Add satellite bands
        for i in range(band_count):

            data[f"B{i + 1}"] = image[i].flatten()

        # Geographic information
        data["Latitude"] = [
            src.xy(row, 0)[1]
            for row in range(height)
            for _ in range(width)
        ]

        data["Longitude"] = [
            src.xy(0, col)[0]
            for _ in range(height)
            for col in range(width)
        ]

    return pd.DataFrame(data)


def create_or_append_excel(
    image_path: str,
    excel_path: str
) -> dict:

    new_data = read_raw_image(image_path)

    excel_path = Path(excel_path)
    excel_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if excel_path.exists():

        old_data = pd.read_excel(excel_path)

        combined = pd.concat(
            [old_data, new_data],
            ignore_index=True
        )

        combined.to_excel(
            excel_path,
            index=False
        )

        operation = "appended"

    else:

        new_data.to_excel(
            excel_path,
            index=False
        )

        combined = new_data
        operation = "created"

    return {
        "status": "success",
        "operation": operation,
        "rows_added": len(new_data),
        "total_rows": len(combined),
        "bands": len(
            [c for c in new_data.columns if c.startswith("B")]
        )
    }