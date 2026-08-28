"""
Spatial validation and exploration-relevant metrics.

The single most important module in this project. Random splits produce
misleadingly high scores because positives cluster spatially — a test
point 200m from a training positive lets the model memorise the map
instead of learning geology.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config


def assign_spatial_blocks(df: pd.DataFrame,
                          block_size_deg: float = config.BLOCK_SIZE_DEG
                          ) -> pd.DataFrame:
    """Tag each row with a coarse lon/lat block id.

    Splits happen at BLOCK level, never at point level, so a positive
    and its close neighbour can never land in different folds.
    """
    df = df.copy()
    df["block_lon"] = (df["longitude"] // block_size_deg).astype(int)
    df["block_lat"] = (df["latitude"] // block_size_deg).astype(int)
    df["block_id"] = (
        df["block_lon"].astype(str) + "_" + df["block_lat"].astype(str)
    )
    return df.drop(columns=["block_lon", "block_lat"])


def spatial_block_folds(df: pd.DataFrame,
                        n_folds: int = config.N_FOLDS,
                        seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Assign whole blocks to folds. Returns df with a 'fold' column."""
    if "block_id" not in df.columns:
        df = assign_spatial_blocks(df)

    blocks = np.array(df["block_id"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(blocks))
    blocks = blocks[perm]
    mapping = {b: i % n_folds for i, b in enumerate(blocks)}

    df = df.copy()
    df["fold"] = df["block_id"].map(mapping)
    return df


def recall_at_top_k(y_true, y_score, k_percent: float = 5.0) -> float:
    """Fraction of known positives captured in the top K% of scores.

    The most exploration-relevant metric: 'if we investigate only the
    top 5% of the area, how many known occurrences do we recover?'
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n_top = max(1, int(len(y_score) * k_percent / 100.0))
    top_idx = np.argsort(y_score)[::-1][:n_top]

    total_positives = y_true.sum()
    if total_positives == 0:
        return float("nan")

    return float(y_true[top_idx].sum() / total_positives)


def evaluate(y_true, y_score, label: str = "") -> dict:
    """Full metric set. PR-AUC is primary — positives are rare, so
    ROC-AUC flatters the model."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    metrics = {
        "model": label,
        "n_test": len(y_true),
        "n_positives": int(y_true.sum()),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_top_1pct": recall_at_top_k(y_true, y_score, 1.0),
        "recall_at_top_5pct": recall_at_top_k(y_true, y_score, 5.0),
        "recall_at_top_10pct": recall_at_top_k(y_true, y_score, 10.0),
    }

    # ROC-AUC reported for completeness but NOT as the headline number
    try:
        metrics["roc_auc_secondary"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc_secondary"] = float("nan")

    return metrics


def print_metrics(m: dict) -> None:
    print(f"\n  {m['model']}")
    print(f"    PR-AUC (primary)     : {m['pr_auc']:.4f}")
    print(f"    Recall @ top 1%      : {m['recall_at_top_1pct']:.3f}")
    print(f"    Recall @ top 5%      : {m['recall_at_top_5pct']:.3f}")
    print(f"    Recall @ top 10%     : {m['recall_at_top_10pct']:.3f}")
    print(f"    ROC-AUC (secondary)  : {m['roc_auc_secondary']:.4f}")
    print(f"    n_test={m['n_test']}, n_positives={m['n_positives']}")


def sanity_check_score(pr_auc: float) -> str:
    """Interpretation guard. In this problem a very high score usually
    means leakage, not success."""
    if pr_auc > 0.90:
        return ("SUSPICIOUS — PR-AUC > 0.90 in MPM almost always means "
                "leakage. Check: banned features present? random split "
                "used somewhere? positives inside active mine footprints?")
    if pr_auc < 0.02:
        return ("TOO LOW — check grid alignment, label coordinates, and "
                "whether features actually sampled non-NaN values.")
    return "PLAUSIBLE — in the expected range for spatial-holdout MPM."
