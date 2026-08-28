"""
Train the full model ladder with spatial cross-validation.

    python -m src.train

Expects data/processed/training_table.parquet (or .csv) with columns:
    target_id, latitude, longitude, <features...>, label
where label = 1 for known positives and 0 for UNLABELED.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from . import config
from .validation import (
    spatial_block_folds, evaluate, print_metrics, sanity_check_score
)
from .pu_learning import BaggingPU

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed — skipping XGB models. pip install xgboost")


def load_table():
    pq = config.DATA_PROCESSED / "training_table.parquet"
    csv = config.DATA_PROCESSED / "training_table.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No training table found. Expected {pq} or {csv}.\n"
        "Run the feature-building step first."
    )


def get_feature_columns(df):
    """Everything that isn't an identifier or the label."""
    exclude = {"target_id", "latitude", "longitude", "label",
               "block_id", "fold", "source"}
    # Layer 3 context columns are pass-through only — never trained on.
    exclude |= set(config.CONTEXT_COLUMNS)
    exclude |= set(config.OUTPUT_ONLY_COLUMNS)
    feats = [c for c in df.columns if c not in exclude]
    config.assert_no_banned(feats)
    return feats


def run_cv(df, feature_cols, model_factory, model_name, is_pu=False):
    """Spatial-block cross-validation. Returns list of per-fold metrics."""
    fold_metrics = []

    for fold in range(config.N_FOLDS):
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold]

        if train["label"].sum() < 3 or test["label"].sum() < 1:
            print(f"    fold {fold}: skipped (too few positives)")
            continue

        X_train = train[feature_cols].astype(float)
        X_test = test[feature_cols].astype(float)

        # impute with TRAIN means only — never let test stats leak in
        train_means = X_train.mean()
        X_train = X_train.fillna(train_means)
        X_test = X_test.fillna(train_means)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        y_train = train["label"].values
        y_test = test["label"].values

        model = model_factory()
        model.fit(X_train_s, y_train)
        scores = model.predict_proba(X_test_s)[:, 1]

        m = evaluate(y_test, scores, label=f"{model_name} (fold {fold})")
        fold_metrics.append(m)

    return fold_metrics


def summarise(fold_metrics, model_name):
    if not fold_metrics:
        return None
    mean_metrics = {
        "model": model_name,
        "pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "pr_auc_std": float(np.std([m["pr_auc"] for m in fold_metrics])),
        "recall_at_top_1pct": float(np.mean(
            [m["recall_at_top_1pct"] for m in fold_metrics])),
        "recall_at_top_5pct": float(np.mean(
            [m["recall_at_top_5pct"] for m in fold_metrics])),
        "recall_at_top_10pct": float(np.mean(
            [m["recall_at_top_10pct"] for m in fold_metrics])),
        "roc_auc_secondary": float(np.mean(
            [m["roc_auc_secondary"] for m in fold_metrics])),
        "n_test": int(np.sum([m["n_test"] for m in fold_metrics])),
        "n_positives": int(np.sum([m["n_positives"] for m in fold_metrics])),
    }
    return mean_metrics


def main():
    df = load_table()
    print(f"Loaded {len(df)} rows")
    print(f"  positives : {int(df['label'].sum())}")
    print(f"  unlabeled : {int((df['label'] == 0).sum())}")

    n_pos = int(df["label"].sum())
    if n_pos < 15:
        print("\nWARNING: fewer than 15 positives. Results will be very "
              "noisy and spatial CV folds may be empty. Consider widening "
              "the study area or adding label sources.")

    feature_cols = get_feature_columns(df)
    print(f"  features  : {len(feature_cols)}")

    # Feature-count discipline: too many features vs positives overfits
    ceiling = max(3, n_pos // 3)
    if len(feature_cols) > ceiling:
        print(f"\nNOTE: {len(feature_cols)} features with {n_pos} positives.")
        print(f"      Rule of thumb ceiling is {ceiling}. Consider dropping "
              "correlated features before trusting these numbers.")

    df = spatial_block_folds(df)
    print(f"  blocks    : {df['block_id'].nunique()}")
    print(f"  folds     : {config.N_FOLDS}\n")

    results = []

    # ---- 1. Baseline: Logistic Regression -------------------------------
    print("[1/4] Logistic Regression (baseline)")
    fm = run_cv(df, feature_cols,
                lambda: LogisticRegression(class_weight="balanced",
                                           max_iter=2000),
                "LogisticRegression")
    s = summarise(fm, "LogisticRegression")
    if s:
        print_metrics(s)
        results.append(s)

    # ---- 2. Random Forest ------------------------------------------------
    print("\n[2/4] Random Forest")
    fm = run_cv(df, feature_cols,
                lambda: RandomForestClassifier(
                    n_estimators=300, class_weight="balanced",
                    random_state=config.RANDOM_SEED, n_jobs=-1),
                "RandomForest")
    s = summarise(fm, "RandomForest")
    if s:
        print_metrics(s)
        results.append(s)

    # ---- 3. XGBoost ------------------------------------------------------
    if HAS_XGB:
        print("\n[3/4] XGBoost")
        fm = run_cv(df, feature_cols,
                    lambda: XGBClassifier(
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="aucpr",
                        random_state=config.RANDOM_SEED),
                    "XGBoost")
        s = summarise(fm, "XGBoost")
        if s:
            print_metrics(s)
            results.append(s)

        # ---- 4. Bagging-PU + XGBoost (final model) ----------------------
        print("\n[4/4] Bagging-PU + XGBoost (final)")
        fm = run_cv(df, feature_cols,
                    lambda: BaggingPU(
                        XGBClassifier(
                            n_estimators=200, max_depth=4,
                            learning_rate=0.05, subsample=0.8,
                            eval_metric="aucpr",
                            random_state=config.RANDOM_SEED),
                        n_estimators=config.PU_N_ESTIMATORS,
                        random_state=config.RANDOM_SEED),
                    "BaggingPU-XGBoost", is_pu=True)
        s = summarise(fm, "BaggingPU-XGBoost")
        if s:
            print_metrics(s)
            results.append(s)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 62)
    print("MODEL LADDER SUMMARY (spatial-block CV)")
    print("=" * 62)
    print(f"{'Model':<24}{'PR-AUC':>10}{'R@5%':>10}{'ROC-AUC':>12}")
    for r in results:
        print(f"{r['model']:<24}{r['pr_auc']:>10.4f}"
              f"{r['recall_at_top_5pct']:>10.3f}"
              f"{r['roc_auc_secondary']:>12.4f}")

    if results:
        best = max(results, key=lambda r: r["pr_auc"])
        print(f"\nBest by PR-AUC: {best['model']} ({best['pr_auc']:.4f})")
        print(f"Sanity: {sanity_check_score(best['pr_auc'])}")

        baseline = next((r for r in results
                         if r["model"] == "LogisticRegression"), None)
        if baseline and best["model"] != "LogisticRegression":
            gain = best["pr_auc"] - baseline["pr_auc"]
            print(f"Gain over baseline: {gain:+.4f}")
            if gain < 0.02:
                print("  -> Marginal. Report honestly that the complex "
                      "model added little over logistic regression.")

    out = config.OUTPUTS / "model_ladder_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
