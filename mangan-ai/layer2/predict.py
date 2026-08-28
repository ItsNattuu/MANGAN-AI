"""
Train final Bagging-PU model, predict, explain with SHAP, and export in
the Layer 2 expected output format.

    python -m src.predict

Outputs
-------
outputs/predictions.csv          all cells, full schema
outputs/ranked_targets.json      per-cell targets (MN-T001 format)
outputs/target_zones.json        aggregated exploration zones
outputs/feature_importance.csv
outputs/shap_summary.png
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from . import config
from .pu_learning import BaggingPU
from .train import load_table, get_feature_columns

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("xgboost required: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("shap not installed — skipping explainability. pip install shap")

TOP_N_TARGETS = 100

# Target class thresholds on the 0-100 prospectivity score
CLASS_HIGH = 70
CLASS_MEDIUM = 40

# Zone aggregation: cells within this radius (degrees) group together.
# ~0.045 deg =~ 5 km at Indian latitudes.
ZONE_EPS_DEG = 0.035   # ~4 km grouping radius
ZONE_MIN_CELLS = 4
ZONE_SCORE_CUTOFF = 98  # only the top 2% of cells seed a zone


def classify_target(score_100: float) -> str:
    """HIGH / MEDIUM / LOW bucket from the 0-100 score."""
    if score_100 >= CLASS_HIGH:
        return "HIGH"
    if score_100 >= CLASS_MEDIUM:
        return "MEDIUM"
    return "LOW"


def confidence_from_uncertainty(unc_series: pd.Series,
                                prob_series: pd.Series) -> pd.Series:
    """Confidence from ensemble disagreement, measured RELATIVE to the
    score itself.

    Raw std is misleading here: a cell scoring 0.98 mechanically has more
    room to vary than one scoring 0.02, so ranking on raw std alone marks
    every top target 'Low'. Dividing by the mean (coefficient of
    variation) asks the fairer question — do the 100 models agree
    *proportionally* about this cell?
    """
    rel = unc_series / prob_series.clip(lower=1e-6)
    q33, q66 = np.percentile(rel, [33, 66])

    def label(r):
        if r <= q33:
            return "High"
        if r <= q66:
            return "Medium"
        return "Low"

    return rel.apply(label)


def build_zones(df: pd.DataFrame) -> list:
    """Aggregate nearby high-scoring cells into exploration target zones
    using density-based clustering on coordinates."""
    hot = df[df["prospectivity_score_100"] >= ZONE_SCORE_CUTOFF].copy()
    if len(hot) < ZONE_MIN_CELLS:
        print(f"  Only {len(hot)} cells above cutoff {ZONE_SCORE_CUTOFF} "
              "— no zones formed.")
        return []

    coords = hot[["latitude", "longitude"]].values
    labels = DBSCAN(eps=ZONE_EPS_DEG,
                    min_samples=ZONE_MIN_CELLS).fit_predict(coords)
    hot = hot.assign(zone=labels)

    zones = []
    zone_counter = 0
    for zid in sorted(set(labels)):
        if zid == -1:  # DBSCAN noise, not a real zone
            continue
        zone_counter += 1
        z = hot[hot["zone"] == zid]
        zones.append({
            "zone_id": f"MN-Z{zone_counter:03d}",
            "n_cells": int(len(z)),
            "centroid_latitude": round(float(z["latitude"].mean()), 6),
            "centroid_longitude": round(float(z["longitude"].mean()), 6),
            "bbox": {
                "lat_min": round(float(z["latitude"].min()), 6),
                "lat_max": round(float(z["latitude"].max()), 6),
                "lon_min": round(float(z["longitude"].min()), 6),
                "lon_max": round(float(z["longitude"].max()), 6),
            },
            "mean_prospectivity_score": round(
                float(z["prospectivity_score_100"].mean()), 1),
            "max_prospectivity_score": round(
                float(z["prospectivity_score_100"].max()), 1),
            "target_class": classify_target(
                float(z["prospectivity_score_100"].mean())),
            "mean_uncertainty": round(float(z["uncertainty"].mean()), 4),
            "top_cell_id": str(z.loc[
                z["prospectivity_score_100"].idxmax(), "target_id"]),
        })

    zones.sort(key=lambda x: -x["mean_prospectivity_score"])
    # renumber after sorting so MN-Z001 is the best zone
    for i, z in enumerate(zones, 1):
        z["zone_id"] = f"MN-Z{i:03d}"
    return zones


def main():
    df = load_table()
    feature_cols = get_feature_columns(df)
    print(f"Training final model on {len(df)} rows, "
          f"{int(df['label'].sum())} positives, {len(feature_cols)} features")

    X = df[feature_cols].astype(float)
    col_means = X.mean()
    X = X.fillna(col_means)
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    y = df["label"].values

    base = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        random_state=config.RANDOM_SEED,
    )
    model = BaggingPU(base, n_estimators=config.PU_N_ESTIMATORS,
                      random_state=config.RANDOM_SEED)
    model.fit(X_s, y)
    print(f"Fitted {len(model.estimators_)} PU estimators")

    # mean = prospectivity probability, std = uncertainty
    probs, uncertainty = model.predict_proba_mean_std(X_s)

    out = df[["latitude", "longitude", "label"]].copy()
    out["prospectivity_probability"] = probs
    out["uncertainty"] = uncertainty
    # percentile rank -> 0-100 score, so it spreads across the full range
    out["prospectivity_score_100"] = (
        out["prospectivity_probability"].rank(pct=True) * 100
    )
    # keep original row position so SHAP rows can be matched back
    out["_orig_idx"] = np.arange(len(out))

    out = out.sort_values("prospectivity_probability", ascending=False)
    out = out.reset_index(drop=True)

    # target IDs in rank order: MN-T001 = highest scoring
    out["target_id"] = [f"MN-T{i + 1:03d}" for i in range(len(out))]
    out["target_class"] = out["prospectivity_score_100"].apply(classify_target)
    out["confidence"] = confidence_from_uncertainty(
        out["uncertainty"], out["prospectivity_probability"])

    # ---- OUTPUT SCHEMA (fixed — Layer 3 depends on this exactly) --------
    # latitude, longitude, Mn_Probability, Mn_Prospectivity,
    # elevation, slope, distance_to_road, distance_to_rail,
    # distance_to_mine, distance_to_power, distance_to_processing,
    # land_use, protected_area, geological_favorability
    out = out.rename(columns={
        "prospectivity_probability": "Mn_Probability",
        "prospectivity_score_100": "Mn_Prospectivity",
    })
    out["Mn_Probability"] = out["Mn_Probability"].round(2)
    out["Mn_Prospectivity"] = out["Mn_Prospectivity"].round(0).astype(int)

    # pull through the source columns the model never saw
    src_cols = ["elevation", "slope"] + config.CONTEXT_COLUMNS \
        + config.OUTPUT_ONLY_COLUMNS
    passthrough = df[src_cols].copy()
    passthrough["_orig_idx"] = np.arange(len(df))
    out = out.merge(passthrough, on="_orig_idx", how="left")

    OUTPUT_SCHEMA = [
        "latitude", "longitude", "Mn_Probability", "Mn_Prospectivity",
        "elevation", "slope",
        "distance_to_road", "distance_to_rail", "distance_to_mine",
        "distance_to_power", "distance_to_processing",
        "land_use", "protected_area", "geological_favorability",
    ]

    final = out[OUTPUT_SCHEMA].copy()
    for c in ["latitude", "longitude"]:
        final[c] = final[c].round(4)
    for c in ["elevation", "distance_to_road", "distance_to_rail",
              "distance_to_mine", "distance_to_power",
              "distance_to_processing"]:
        final[c] = final[c].round(1)
    final["slope"] = final["slope"].round(1)
    final["geological_favorability"] = final["geological_favorability"].round(2)

    final.to_csv(config.OUTPUTS / "predictions.csv", index=False)
    print(f"Saved -> {config.OUTPUTS / 'predictions.csv'}")
    print(f"  {len(final)} rows x {len(OUTPUT_SCHEMA)} columns")

    # ---- feature importance ---------------------------------------------
    imp = model.feature_importances_mean()
    if imp is not None:
        imp_df = pd.DataFrame({
            "feature": feature_cols, "importance": imp
        }).sort_values("importance", ascending=False)
        imp_df.to_csv(config.OUTPUTS / "feature_importance.csv", index=False)
        print("\nTop 8 features:")
        for _, r in imp_df.head(8).iterrows():
            print(f"  {r['feature']:<28}{r['importance']:.4f}")

    # ---- SHAP ------------------------------------------------------------
    top_factors = {}
    if HAS_SHAP:
        print("\nComputing SHAP values...")
        explainer = shap.TreeExplainer(model.estimators_[0])
        top_orig = out["_orig_idx"].values[:TOP_N_TARGETS]
        shap_vals = explainer.shap_values(X_s[top_orig])

        for rank in range(len(top_orig)):
            contribs = dict(zip(feature_cols, shap_vals[rank]))
            top = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))[:4]
            top_factors[rank] = [
                {"feature": f, "contribution": round(float(v), 4)}
                for f, v in top
            ]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_vals, X.iloc[top_orig],
                              feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(config.OUTPUTS / "shap_summary.png",
                        dpi=140, bbox_inches="tight")
            plt.close()
            print(f"Saved -> {config.OUTPUTS / 'shap_summary.png'}")
        except Exception as e:
            print(f"  (plot skipped: {e})")

    # ---- console preview -------------------------------------------------
    print("\n" + "=" * 100)
    print("OUTPUT PREVIEW — first 10 rows")
    print("=" * 100)
    with pd.option_context("display.width", 200,
                           "display.max_columns", 20):
        print(final.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
