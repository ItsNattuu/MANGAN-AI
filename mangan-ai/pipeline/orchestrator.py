from pathlib import Path
import pandas as pd

from layer1.raw_2_excel import read_raw_image
from layer1.manganese_features import calculate_manganese_features

from layer4.agent.agent import analyze


def run_layer1(image_path):

    print("\n========== LAYER 1 ==========")

    df = read_raw_image(
        image_path
    )

    df = calculate_manganese_features(
        df
    )

    print(
        f"Layer 1 processed {len(df)} pixels."
    )

    return df


def run_layer2(layer1_df):

    print("\n========== LAYER 2 ==========")

    # IMPORTANT:
    # Your actual Layer 2 model currently
    # expects its trained feature table.
    #
    # This adapter is where the Layer 1
    # dataframe is converted into the
    # exact schema Layer 2 expects.

    from layer2.predict import run_prediction_from_dataframe

    result = run_prediction_from_dataframe(
        layer1_df
    )

    print(
        f"Layer 2 generated {len(result)} targets."
    )

    return result


def run_layer3(layer2_df):

    print("\n========== LAYER 3 ==========")

    from layer3.adapter import run_layer3_from_layer2

    result = run_layer3_from_layer2(
        layer2_df
    )

    print(
        f"Layer 3 generated {len(result)} targets."
    )

    return result


def run_layer4(
    request,
    layer2_results,
    layer3_results
):

    print("\n========== LAYER 4 ==========")

    result = analyze(
        request=request,
        layer2_results=layer2_results,
        layer3_results=layer3_results
    )

    return result


def run_pipeline(
    image_path,
    request="Find the best manganese exploration targets"
):

    # -----------------------------------
    # LAYER 1
    # -----------------------------------

    layer1_df = run_layer1(
        image_path
    )

    # -----------------------------------
    # LAYER 2
    # -----------------------------------

    layer2_df = run_layer2(
        layer1_df
    )

    # Convert to records for Layer 4
    layer2_results = (
        layer2_df
        .to_dict(orient="records")
    )

    # -----------------------------------
    # LAYER 3
    # -----------------------------------

    layer3_df = run_layer3(
        layer2_df
    )

    layer3_results = (
        layer3_df
        .to_dict(orient="records")
    )

    # -----------------------------------
    # LAYER 4
    # -----------------------------------

    layer4_result = run_layer4(
        request=request,
        layer2_results=layer2_results,
        layer3_results=layer3_results
    )

    return {
        "layer1": {
            "rows_processed": len(layer1_df)
        },

        "layer2": layer2_results,

        "layer3": layer3_results,

        "layer4": layer4_result
    }
