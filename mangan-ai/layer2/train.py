"""
Train the Layer 2 model ladder.

Prototype workflow:

```
Layer 1 processed data
        ↓
training_table.csv / parquet
        ↓
spatial folds
        ↓
Logistic Regression
        ↓
Random Forest
        ↓
XGBoost
        ↓
Bagging-PU + XGBoost
        ↓
best model / prospectivity output
```

## IMPORTANT

This project currently uses synthetic/fake data for demonstration.
Therefore the resulting metrics are prototype/demo metrics and
must NOT be presented as real-world geological validation.
"""

import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config
from .pu_learning import BaggingPU
from .validation import (
evaluate,
print_metrics,
sanity_check_score,
spatial_block_folds,
)

try:
from xgboost import XGBClassifier

```
HAS_XGB = True
```

except ImportError:

```
HAS_XGB = False

print(
    "WARNING: xgboost is not installed. "
    "XGBoost models will be skipped."
)
```

# =========================================================

# DATA LOADING

# =========================================================

def load_table():
"""
Load the Layer 2 training table.

```
Preferred order:
    1. parquet
    2. csv
"""

parquet_path = (
    config.DATA_PROCESSED
    / "training_table.parquet"
)

csv_path = (
    config.DATA_PROCESSED
    / "training_table.csv"
)

if parquet_path.exists():

    print(
        f"Loading training table: {parquet_path}"
    )

    return pd.read_parquet(
        parquet_path
    )

if csv_path.exists():

    print(
        f"Loading training table: {csv_path}"
    )

    return pd.read_csv(
        csv_path
    )

raise FileNotFoundError(
    "\nNo Layer 2 training table found.\n\n"
    f"Expected either:\n"
    f"  {parquet_path}\n"
    f"  {csv_path}\n\n"
    "Create the prototype training table first."
)
```

# =========================================================

# VALIDATION OF INPUT TABLE

# =========================================================

def validate_training_table(df):
"""
Check that the input table contains the minimum
columns required by Layer 2.
"""

```
required = {
    "label",
    "latitude",
    "longitude",
}

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:

    raise ValueError(
        "Layer 2 training table is missing "
        f"required columns: {missing}"
    )

if df.empty:

    raise ValueError(
        "Layer 2 training table is empty."
    )

# -----------------------------------------------------
# Label validation
# -----------------------------------------------------

labels = set(
    pd.Series(
        df["label"]
    )
    .dropna()
    .unique()
)

invalid_labels = labels - {0, 1}

if invalid_labels:

    raise ValueError(
        "The 'label' column must contain only "
        f"0 and 1. Found: {invalid_labels}"
    )
```

# =========================================================

# FEATURE SELECTION

# =========================================================

def get_feature_columns(df):
"""
Select only Layer 2 ML features.

```
Context columns are deliberately excluded because
Layer 3 needs them later for mining-feasibility
analysis.
"""

exclude = {
    "target_id",
    "latitude",
    "longitude",
    "label",
    "block_id",
    "fold",
    "source",
}

# Layer 3 context is passed through but NEVER trained on.
exclude.update(
    config.CONTEXT_COLUMNS
)

# Derived/output-only fields are also excluded.
exclude.update(
    config.OUTPUT_ONLY_COLUMNS
)

# -----------------------------------------------------
# Prefer the explicitly defined Layer 2 feature list.
# -----------------------------------------------------

configured_features = [
    feature
    for feature in config.all_features()
    if feature in df.columns
]

# -----------------------------------------------------
# If the table contains configured features, use them.
# This prevents accidental training on arbitrary columns.
# -----------------------------------------------------

if configured_features:

    feature_columns = configured_features

else:

    # Fallback for the synthetic prototype.
    feature_columns = [
        column
        for column in df.columns
        if column not in exclude
    ]

if not feature_columns:

    raise ValueError(
        "No usable Layer 2 features were found "
        "in the training table."
    )

# -----------------------------------------------------
# Leakage protection
# -----------------------------------------------------

config.assert_no_banned(
    feature_columns
)

return feature_columns
```

# =========================================================

# FEATURE PREPARATION

# =========================================================

def prepare_features(
train,
test,
feature_columns,
):
"""
Prepare train/test matrices.

```
Numeric columns:
    median imputation

Categorical columns:
    encoded using pandas category codes

Scaling:
    StandardScaler

IMPORTANT:
    Statistics are fitted using TRAINING data only.
"""

X_train = train[
    feature_columns
].copy()

X_test = test[
    feature_columns
].copy()

# -----------------------------------------------------
# Convert categorical columns
# -----------------------------------------------------

for column in feature_columns:

    if column in config.CATEGORICAL_FEATURES:

        combined = pd.concat(
            [
                X_train[column],
                X_test[column],
            ],
            axis=0
        ).astype("category")

        categories = combined.cat.categories

        X_train[column] = (
            pd.Categorical(
                X_train[column],
                categories=categories
            ).codes
        )

        X_test[column] = (
            pd.Categorical(
                X_test[column],
                categories=categories
            ).codes
        )

# -----------------------------------------------------
# Force numeric representation
# -----------------------------------------------------

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

# -----------------------------------------------------
# Replace infinities
# -----------------------------------------------------

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# -----------------------------------------------------
# Median imputation
# -----------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_test = imputer.transform(
    X_test
)

# -----------------------------------------------------
# Standardization
# -----------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

return (
    X_train_scaled,
    X_test_scaled,
)
```

# =========================================================

# SPATIAL CROSS VALIDATION

# =========================================================

def run_cv(
df,
feature_columns,
model_factory,
model_name,
):
"""
Run spatial-block cross-validation.

```
Each fold is trained only on other spatial blocks.
"""

fold_metrics = []

for fold in range(
    config.N_FOLDS
):

    train = df[
        df["fold"] != fold
    ].copy()

    test = df[
        df["fold"] == fold
    ].copy()

    # -------------------------------------------------
    # Check whether the fold has enough positives.
    # -------------------------------------------------

    train_positive_count = int(
        train["label"].sum()
    )

    test_positive_count = int(
        test["label"].sum()
    )

    if (
        train_positive_count < 3
        or test_positive_count < 1
    ):

        print(
            f"    fold {fold}: skipped "
            "(too few positive samples)"
        )

        continue

    # -------------------------------------------------
    # Prepare features
    # -------------------------------------------------

    X_train, X_test = prepare_features(
        train,
        test,
        feature_columns,
    )

    y_train = train[
        "label"
    ].astype(int).values

    y_test = test[
        "label"
    ].astype(int).values

    # -------------------------------------------------
    # Train model
    # -------------------------------------------------

    model = model_factory()

    model.fit(
        X_train,
        y_train
    )

    # -------------------------------------------------
    # Predict probability of prospectivity
    # -------------------------------------------------

    scores = model.predict_proba(
        X_test
    )[:, 1]

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------

    metrics = evaluate(
        y_test,
        scores,
        label=f"{model_name} (fold {fold})",
    )

    fold_metrics.append(
        metrics
    )

return fold_metrics
```

# =========================================================

# METRIC SUMMARY

# =========================================================

def summarise(
fold_metrics,
model_name,
):
"""
Calculate mean cross-validation metrics.
"""

```
if not fold_metrics:
    return None

return {
    "model": model_name,

    "pr_auc": float(
        np.mean(
            [
                m["pr_auc"]
                for m in fold_metrics
            ]
        )
    ),

    "pr_auc_std": float(
        np.std(
            [
                m["pr_auc"]
                for m in fold_metrics
            ]
        )
    ),

    "recall_at_top_1pct": float(
        np.mean(
            [
                m["recall_at_top_1pct"]
                for m in fold_metrics
            ]
        )
    ),

    "recall_at_top_5pct": float(
        np.mean(
            [
                m["recall_at_top_5pct"]
                for m in fold_metrics
            ]
        )
    ),

    "recall_at_top_10pct": float(
        np.mean(
            [
                m["recall_at_top_10pct"]
                for m in fold_metrics
            ]
        )
    ),

    "roc_auc_secondary": float(
        np.mean(
            [
                m["roc_auc_secondary"]
                for m in fold_metrics
            ]
        )
    ),

    "n_test": int(
        np.sum(
            [
                m["n_test"]
                for m in fold_metrics
            ]
        )
    ),

    "n_positives": int(
        np.sum(
            [
                m["n_positives"]
                for m in fold_metrics
            ]
        )
    ),
}
```

# =========================================================

# MODEL FACTORIES

# =========================================================

def logistic_regression_factory():
return LogisticRegression(
class_weight="balanced",
max_iter=2000,
random_state=config.RANDOM_SEED,
)

def random_forest_factory():
return RandomForestClassifier(
n_estimators=300,
class_weight="balanced",
random_state=config.RANDOM_SEED,
n_jobs=-1,
)

def xgboost_factory():
return XGBClassifier(
n_estimators=300,
max_depth=4,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
eval_metric="aucpr",
random_state=config.RANDOM_SEED,
)

# =========================================================

# MAIN TRAINING PIPELINE

# =========================================================

def main():

```
print("\n" + "=" * 62)
print("MANGAN-AI — LAYER 2")
print("Mineral Prospectivity Model Training")
print("=" * 62)

# -----------------------------------------------------
# 1. Load data
# -----------------------------------------------------

df = load_table()

print(
    f"\nLoaded {len(df)} rows."
)

# -----------------------------------------------------
# 2. Validate table
# -----------------------------------------------------

validate_training_table(
    df
)

# -----------------------------------------------------
# 3. Label statistics
# -----------------------------------------------------

positive_count = int(
    df["label"].sum()
)

unlabeled_count = int(
    (df["label"] == 0).sum()
)

print(
    f"  positives : {positive_count}"
)

print(
    f"  unlabeled : {unlabeled_count}"
)

if positive_count < 15:

    print(
        "\nWARNING:"
    )

    print(
        "The prototype contains fewer than "
        "15 positive samples."
    )

    print(
        "Spatial validation may therefore skip "
        "some folds."
    )

# -----------------------------------------------------
# 4. Select features
# -----------------------------------------------------

feature_columns = get_feature_columns(
    df
)

print(
    f"\nFeatures used for Layer 2: "
    f"{len(feature_columns)}"
)

for feature in feature_columns:

    print(
        f"  - {feature}"
    )

# -----------------------------------------------------
# 5. Feature-count warning
# -----------------------------------------------------

ceiling = max(
    3,
    positive_count // 3
)

if len(feature_columns) > ceiling:

    print(
        "\nWARNING:"
    )

    print(
        f"{len(feature_columns)} features "
        f"with {positive_count} positives."
    )

    print(
        f"Prototype rule-of-thumb ceiling: "
        f"{ceiling}"
    )

# -----------------------------------------------------
# 6. Create spatial folds
# -----------------------------------------------------

df = spatial_block_folds(
    df
)

print(
    f"\nSpatial blocks: "
    f"{df['block_id'].nunique()}"
)

print(
    f"CV folds: "
    f"{config.N_FOLDS}"
)

# -----------------------------------------------------
# 7. Model ladder
# -----------------------------------------------------

results = []

# -----------------------------------------------------
# Model 1 — Logistic Regression
# -----------------------------------------------------

print(
    "\n[1/4] Logistic Regression"
)

fold_metrics = run_cv(
    df,
    feature_columns,
    logistic_regression_factory,
    "LogisticRegression",
)

summary = summarise(
    fold_metrics,
    "LogisticRegression",
)

if summary:

    print_metrics(
        summary
    )

    results.append(
        summary
    )

# -----------------------------------------------------
# Model 2 — Random Forest
# -----------------------------------------------------

print(
    "\n[2/4] Random Forest"
)

fold_metrics = run_cv(
    df,
    feature_columns,
    random_forest_factory,
    "RandomForest",
)

summary = summarise(
    fold_metrics,
    "RandomForest",
)

if summary:

    print_metrics(
        summary
    )

    results.append(
        summary
    )

# -----------------------------------------------------
# Models 3 and 4 require XGBoost
# -----------------------------------------------------

if HAS_XGB:

    # -------------------------------------------------
    # Model 3 — XGBoost
    # -------------------------------------------------

    print(
        "\n[3/4] XGBoost"
    )

    fold_metrics = run_cv(
        df,
        feature_columns,
        xgboost_factory,
        "XGBoost",
    )

    summary = summarise(
        fold_metrics,
        "XGBoost",
    )

    if summary:

        print_metrics(
            summary
        )

        results.append(
            summary
        )

    # -------------------------------------------------
    # Model 4 — Bagging PU + XGBoost
    # -------------------------------------------------

    print(
        "\n[4/4] Bagging-PU + XGBoost"
    )

    def pu_factory():

        return BaggingPU(
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                eval_metric="aucpr",
                random_state=config.RANDOM_SEED,
            ),
            n_estimators=(
                config.PU_N_ESTIMATORS
            ),
            random_state=(
                config.RANDOM_SEED
            ),
        )

    fold_metrics = run_cv(
        df,
        feature_columns,
        pu_factory,
        "BaggingPU-XGBoost",
    )

    summary = summarise(
        fold_metrics,
        "BaggingPU-XGBoost",
    )

    if summary:

        print_metrics(
            summary
        )

        results.append(
            summary
        )

else:

    print(
        "\nXGBoost unavailable."
    )

    print(
        "Continuing with Logistic Regression "
        "and Random Forest."
    )

# =====================================================
# MODEL LADDER SUMMARY
# =====================================================

print(
    "\n" + "=" * 62
)

print(
    "MODEL LADDER SUMMARY"
)

print(
    "=" * 62
)

if not results:

    print(
        "\nNo model completed successfully."
    )

    print(
        "Check the number of positive samples "
        "and spatial folds."
    )

    return

print(
    f"{'Model':<24}"
    f"{'PR-AUC':>10}"
    f"{'R@5%':>10}"
    f"{'ROC-AUC':>12}"
)

for result in results:

    print(
        f"{result['model']:<24}"
        f"{result['pr_auc']:>10.4f}"
        f"{result['recall_at_top_5pct']:>10.3f}"
        f"{result['roc_auc_secondary']:>12.4f}"
    )

# -----------------------------------------------------
# Best model
# -----------------------------------------------------

best = max(
    results,
    key=lambda result: result["pr_auc"],
)

print(
    f"\nBest prototype model by PR-AUC: "
    f"{best['model']}"
)

print(
    f"PR-AUC: "
    f"{best['pr_auc']:.4f}"
)

print(
    "Sanity:",
    sanity_check_score(
        best["pr_auc"]
    ),
)

# -----------------------------------------------------
# Compare with baseline
# -----------------------------------------------------

baseline = next(
    (
        result
        for result in results
        if result["model"]
        == "LogisticRegression"
    ),
    None,
)

if (
    baseline
    and best["model"]
    != "LogisticRegression"
):

    gain = (
        best["pr_auc"]
        - baseline["pr_auc"]
    )

    print(
        f"Gain over baseline: "
        f"{gain:+.4f}"
    )

# -----------------------------------------------------
# Save results
# -----------------------------------------------------

output_path = (
    config.OUTPUTS
    / "model_ladder_results.json"
)

output_path.write_text(
    json.dumps(
        results,
        indent=2
    )
)

print(
    f"\nSaved model results -> "
    f"{output_path}"
)

print(
    "\nNOTE:"
)

print(
    "These metrics come from the current "
    "prototype/synthetic dataset."
)

print(
    "They are for demonstrating the MANGAN-AI "
    "workflow, not for claiming real geological "
    "model performance."
)
```

if **name** == "**main**":
main()
