from manganese_identifier import identify_manganese


IMAGE = "data/raw/sample.tif"

OUTPUT = "data/processed/manganese_dataset.xlsx"


result = identify_manganese(
    image_path=IMAGE,
    output_excel=OUTPUT
)


print("\n==============================")
print("MANGAN-AI RESULT")
print("==============================")

for key, value in result.items():

    print(
        f"{key}: {value}"
    )
