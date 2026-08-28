from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

# ---------------------------------------------------------

# Band names used by the MANGAN-AI pipeline

# ---------------------------------------------------------

STANDARD_BANDS = [
"B01",
"B02",
"B03",
"B04",
"B05",
"B06",
"B07",
"B08",
"B09",
"B10",
"B11",
"B12",
"B13",
]

def read_raw_image(image_path: str) -> pd.DataFrame:
"""
Read a multispectral raster and convert its pixels
into a Pandas DataFrame.

```
Each row represents one pixel.

Output columns include:

    Pixel_X
    Pixel_Y
    B01, B02, ...
    Latitude
    Longitude

Geographic coordinates are returned in EPSG:4326.
"""

# -----------------------------------------------------
# 1. Validate input path
# -----------------------------------------------------

image_path = Path(image_path)

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )

# -----------------------------------------------------
# 2. Open raster
# -----------------------------------------------------

with rasterio.open(image_path) as src:

    image = src.read()

    width = src.width
    height = src.height
    band_count = src.count

    # -------------------------------------------------
    # 3. Pixel coordinates
    # -------------------------------------------------

    x, y = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    data = {
        "Pixel_X": x.flatten(),
        "Pixel_Y": y.flatten(),
    }

    # -------------------------------------------------
    # 4. Satellite bands
    # -------------------------------------------------

    for i in range(band_count):

        if i < len(STANDARD_BANDS):
            band_name = STANDARD_BANDS[i]
        else:
            # Support rasters with more bands than
            # the standard prototype band list.
            band_name = f"B{i + 1:02d}"

        data[band_name] = image[i].flatten()

    # -------------------------------------------------
    # 5. Geographic coordinates
    # -------------------------------------------------

    rows, cols = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing="ij"
    )

    rows_flat = rows.flatten()
    cols_flat = cols.flatten()

    xs, ys = rasterio.transform.xy(
        src.transform,
        rows_flat,
        cols_flat,
        offset="center"
    )

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # -------------------------------------------------
    # 6. Convert to latitude / longitude
    # -------------------------------------------------

    if src.crs is not None:

        longitude, latitude = transform(
            src.crs,
            "EPSG:4326",
            xs.tolist(),
            ys.tolist()
        )

        data["Latitude"] = latitude
        data["Longitude"] = longitude

    else:

        # -------------------------------------------------
        # Fallback for synthetic rasters with no CRS.
        #
        # We keep the raster coordinates rather than
        # silently pretending they are real GPS values.
        # -------------------------------------------------

        data["Latitude"] = ys
        data["Longitude"] = xs

# -----------------------------------------------------
# 7. Create DataFrame
# -----------------------------------------------------

df = pd.DataFrame(data)

return df
```

def create_or_append_excel(
image_path: str,
excel_path: str
) -> dict:
"""
Read a raster and create or append its pixel data
to an Excel file.

```
This helper is retained for backwards compatibility
with the existing Layer 1 workflow.
"""

# -----------------------------------------------------
# Read new raster data
# -----------------------------------------------------

new_data = read_raw_image(image_path)

# -----------------------------------------------------
# Prepare output path
# -----------------------------------------------------

excel_path = Path(excel_path)

excel_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

# -----------------------------------------------------
# Append if the file already exists
# -----------------------------------------------------

if excel_path.exists():

    old_data = pd.read_excel(
        excel_path
    )

    combined = pd.concat(
        [old_data, new_data],
        ignore_index=True
    )

    combined.to_excel(
        excel_path,
        index=False
    )

    operation = "appended"

# -----------------------------------------------------
# Otherwise create a new file
# -----------------------------------------------------

else:

    new_data.to_excel(
        excel_path,
        index=False
    )

    combined = new_data

    operation = "created"

# -----------------------------------------------------
# Return summary
# -----------------------------------------------------

band_columns = [
    column
    for column in new_data.columns
    if column.startswith("B")
    and column[1:].isdigit()
]

return {
    "status": "success",
    "operation": operation,
    "rows_added": len(new_data),
    "total_rows": len(combined),
    "bands": len(band_columns)
}
```
