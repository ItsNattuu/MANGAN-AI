from pathlib import Path

from .manganese_identifier import identify_manganese

# ---------------------------------------------------------

# Layer 1 directory

# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------

# Input / output paths

# ---------------------------------------------------------

IMAGE = BASE_DIR / "data" / "raw" / "sample.tif"

OUTPUT = (
BASE_DIR
/ "data"
/ "processed"
/ "manganese_dataset.xlsx"
)

# ---------------------------------------------------------

# Run Layer 1

# ---------------------------------------------------------

if **name** == "**main**":

```
result = identify_manganese(
    image_path=str(IMAGE),
    output_excel=str(OUTPUT)
)

# -----------------------------------------------------
# Display result
# -----------------------------------------------------

print("\n==============================")
print("MANGAN-AI - LAYER 1 RESULT")
print("==============================")

for key, value in result.items():
    print(f"{key}: {value}")
```
