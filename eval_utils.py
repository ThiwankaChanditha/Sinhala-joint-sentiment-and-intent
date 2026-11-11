"""
Shared Evaluation, Preprocessing, and Split Utilities for Sinhala Joint Sentiment & Intent Baselines.

Provides:
- Canonical label definitions (4-class Sentiment, 6-class Intent)
- Preprocessing functions for Run A (clean ZWJ) and Run B (emoji + conversational punctuation kept)
- Deterministic split loader (`splits_runA.csv`, `splits_runB.csv`) with strict leakage assertions
- Unified evaluation metrics:
    * Sent Acc, Sent Macro-F1 (4-cls), Sent Macro-F1 (3-cls: Negative/Neutral/Positive)
    * Intent Acc, Intent Macro-F1 (6-cls)
    * Joint Acc (both tasks correct)
    * Per-class Precision / Recall / F1 / Support for both tasks
    * Intent confusion matrix exported as CSV and PNG
    * Per-run JSON export (`results/{notebook}_{run}_{seed}.json`)
    * Per-row test prediction export (`preds/{model}_{run}_{seed}.csv`)
"""

import os
import re
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Canonical Label Sets (Alphabetical, matching sklearn LabelEncoder) ────────
SENT_CLASSES = ["Conflict", "Negative", "Neutral", "Positive"]
INTENT_CLASSES = ["complaint", "feedback", "greeting", "inquiry", "other", "suggestion"]

NS = len(SENT_CLASSES)      # 4
NI = len(INTENT_CLASSES)    # 6

# 3-class sentiment indices excluding Conflict (index 0): Negative (1), Neutral (2), Positive (3)
SENT_3CLS_INDICES = [1, 2, 3]
SENT_3CLS_NAMES = ["Negative", "Neutral", "Positive"]

SEEDS = [42, 43, 44]

SENT_TO_ID = {c.lower(): i for i, c in enumerate(SENT_CLASSES)}
ID_TO_SENT = {i: c for i, c in enumerate(SENT_CLASSES)}

INTENT_TO_ID = {c.lower(): i for i, c in enumerate(INTENT_CLASSES)}
ID_TO_INTENT = {i: c for i, c in enumerate(INTENT_CLASSES)}


def log_environment() -> Dict[str, str]:
    """Log Python and key library versions for reproducibility."""
    import sys
    import sklearn
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not_installed"
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = "not_installed"

    print("Environment versions:", json.dumps(info, indent=2))
    return info


def set_all_seeds(seed: int) -> None:
    """Fix random seeds across random, numpy, and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Path Resolution ───────────────────────────────────────────────────────────
def resolve_path(filename: str, custom_path: Optional[str] = None) -> str:
    """Resolve dataset or split file paths across local workspace, Colab, and Kaggle."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if os.path.exists(filename):
        return filename

    candidates = [
        os.path.join(".", filename),
        os.path.join("data", "processed", filename),
        os.path.join("data", "benchmark", filename),
        os.path.join("Sinhala-joint-sentiment-and-intent", "data", "processed", filename),
        os.path.join("Sinhala-joint-sentiment-and-intent", "data", "benchmark", filename),
        os.path.join("..", filename),
        os.path.join("..", "data", "processed", filename),
        os.path.join("..", "data", "benchmark", filename),
        os.path.join("..", "..", filename),
        os.path.join("..", "..", "data", "processed", filename),
        os.path.join("..", "..", "data", "benchmark", filename),
        f"/content/{filename}",
        f"/content/Sinhala-joint-sentiment-and-intent/{filename}",
        f"/content/Sinhala-joint-sentiment-and-intent/data/processed/{filename}",
        f"/content/Sinhala-joint-sentiment-and-intent/data/benchmark/{filename}",
        f"/kaggle/working/{filename}",
        f"/kaggle/input/datasets/thiwankachanditha/sold-data-sentiment-intent/{filename}",
        f"/kaggle/input/datasets/thiwankachanditha/sentiment-intent-gold-labeled/{filename}",
    ]
    # Note: annotation_with_sentiment_intent.csv is also named sinhala_gold_benchmark_300.csv in the repo
    if os.path.basename(filename) == "annotation_with_sentiment_intent.csv":
        candidates.extend([
            os.path.join("data", "benchmark", "sinhala_gold_benchmark_300.csv"),
            os.path.join("Sinhala-joint-sentiment-and-intent", "data", "benchmark", "sinhala_gold_benchmark_300.csv"),
            os.path.join("..", "data", "benchmark", "sinhala_gold_benchmark_300.csv"),
            os.path.join("..", "..", "data", "benchmark", "sinhala_gold_benchmark_300.csv"),
            "/content/sinhala_gold_benchmark_300.csv",
            "/content/Sinhala-joint-sentiment-and-intent/data/benchmark/sinhala_gold_benchmark_300.csv",
        ])

    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not locate '{filename}'. Checked: {candidates}")


# ── Text Preprocessing ────────────────────────────────────────────────────────
def clean_text_runA(text: str) -> str:
    """
    Run A preprocessing (matches `mlm-pooling-ablation (1).ipynb`):
    - Lowercase and strip URLs first
    - Keep word chars, whitespace, Sinhala block (U+0D80-U+0DFF), and ZWJ/ZWNJ (U+200C/U+200D)
    """
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s\u0D80-\u0DFF\u200c\u200d]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text_runB(text: str) -> str:
    """
    Run B preprocessing (matches `mlm_pooling_ablation_proper.ipynb`):
    - Lowercase and strip URLs first
    - Keep word chars, whitespace, Sinhala block, ZWJ/ZWNJ, conversational punctuation (?!.), and emojis
    """
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(
        r"[^\w\s\u0D80-\u0DFF\u200c\u200d?!.\U0001F300-\U0001FAFF\u2600-\u27BF]",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, run: str = "runA") -> str:
    """Dispatch text cleaning based on run identifier ('runA' or 'runB')."""
    r = run.strip().lower()
    if r in ("runa", "a", "run_a"):
        return clean_text_runA(text)
    elif r in ("runb", "b", "run_b"):
        return clean_text_runB(text)
    else:
        raise ValueError(f"Unknown run '{run}'. Expected 'runA' or 'runB'.")


# ── Label Encoding Helpers ────────────────────────────────────────────────────
def encode_sentiment(series: pd.Series) -> np.ndarray:
    return np.array([SENT_TO_ID[str(v).strip().lower()] for v in series], dtype=np.int64)


def encode_intent(series: pd.Series) -> np.ndarray:
    return np.array([INTENT_TO_ID[str(v).strip().lower()] for v in series], dtype=np.int64)


# ── Split Generation & Loading ────────────────────────────────────────────────
def _safe_split(d: pd.DataFrame, size, key: str, seed: int = 42):
    from sklearn.model_selection import train_test_split
    ok = d[key].value_counts().min() >= 2
    return train_test_split(d, test_size=size, random_state=seed, stratify=d[key] if ok else None)


def export_splits(
    silver_csv: Optional[str] = None,
    gold_csv: Optional[str] = None,
    out_dir: str = ".",
) -> Tuple[str, str]:
    """
    Export exact train/val/test UIDs matching `mlm-pooling-ablation (1).ipynb` (Run A)
    and `mlm_pooling_ablation_proper.ipynb` (Run B) to `splits_runA.csv` and `splits_runB.csv`.
    """
    silver_path = resolve_path("sinhala_silver_labels.csv", silver_csv)
    gold_path = resolve_path("annotation_with_sentiment_intent.csv", gold_csv)

    raw_df = pd.read_csv(silver_path)
    raw_df = raw_df.rename(
        columns={"Sentence": "raw_text", "intent": "intent_label", "Sentiment": "sentiment_label"}
    )
    raw_df.insert(0, "uid", np.arange(len(raw_df)))
    raw_df = raw_df[["uid", "raw_text", "sentiment_label", "intent_label"]].dropna().copy()

    gold_df = pd.read_csv(gold_path, keep_default_na=False)
    gold_uids = set(gold_df["uid"].astype(int))

    # --- Run A ---
    df_a = raw_df.copy()
    df_a["text"] = df_a["raw_text"].apply(clean_text_runA)
    df_a = df_a[df_a["text"].str.len() > 3]
    df_a = df_a.drop_duplicates("text").reset_index(drop=True)
    df_a["sent_id"] = encode_sentiment(df_a["sentiment_label"])
    df_a["intent_id"] = encode_intent(df_a["intent_label"])

    pool_a, test_a = _safe_split(df_a, 300, "intent_id", 42)
    train_a, val_a = _safe_split(pool_a, 0.15, "intent_id", 42)

    assert len(train_a) == 4609 and len(val_a) == 814 and len(test_a) == 300
    assert set(test_a["uid"]) == gold_uids, "Run A test UIDs must match the 300 gold UIDs"

    split_a_df = pd.concat([
        train_a[["uid"]].assign(split="train"),
        val_a[["uid"]].assign(split="val"),
        test_a[["uid"]].assign(split="test"),
    ], ignore_index=True)

    path_a = os.path.join(out_dir, "splits_runA.csv")
    split_a_df.to_csv(path_a, index=False)

    # --- Run B ---
    df_b = raw_df.copy()
    df_b["is_gold"] = df_b["uid"].isin(gold_uids)
    df_b["text"] = df_b["raw_text"].apply(clean_text_runB)
    df_b = df_b[(df_b["text"].str.len() > 3) | df_b["is_gold"]]

    g_b, rest_b = df_b[df_b["is_gold"]], df_b[~df_b["is_gold"]]
    rest_b = rest_b[~rest_b["text"].isin(set(g_b["text"]))].drop_duplicates("text")
    df_b = pd.concat([g_b, rest_b]).sort_values("uid").reset_index(drop=True)
    df_b["sent_id"] = encode_sentiment(df_b["sentiment_label"])
    df_b["intent_id"] = encode_intent(df_b["intent_label"])

    test_b = df_b[df_b["uid"].isin(gold_uids)].copy()
    pool_b = df_b[~df_b["uid"].isin(gold_uids)].copy()
    train_b, val_b = _safe_split(pool_b, 0.15, "intent_id", 42)

    assert len(train_b) == 4703 and len(val_b) == 831 and len(test_b) == 300
    assert set(test_b["uid"]) == gold_uids, "Run B test UIDs must match the 300 gold UIDs"

    split_b_df = pd.concat([
        train_b[["uid"]].assign(split="train"),
        val_b[["uid"]].assign(split="val"),
        test_b[["uid"]].assign(split="test"),
    ], ignore_index=True)

    path_b = os.path.join(out_dir, "splits_runB.csv")
    split_b_df.to_csv(path_b, index=False)

    return path_a, path_b


def load_splits(
    run: str = "runA",
    silver_csv: Optional[str] = None,
    gold_csv: Optional[str] = None,
    split_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-exported train, val, and test DataFrames for `runA` or `runB` without re-splitting.

    Returns:
        train_df, val_df, test_df
        - `train_df` and `val_df` contain human `sent_id` and silver `intent_id`.
        - `test_df` contains human `sent_id` (`gold_sent_id`) and human `gold_intent_id` (plus silver `silver_intent_id` for reference).
    """
    r = run.strip().lower()
    is_run_a = r in ("runa", "a", "run_a")
    default_split_file = "splits_runA.csv" if is_run_a else "splits_runB.csv"

    try:
        split_path = resolve_path(default_split_file, split_csv)
    except FileNotFoundError:
        print(f"Split file {default_split_file} not found; exporting splits from ablation recipe...")
        export_splits(silver_csv=silver_csv, gold_csv=gold_csv, out_dir=".")
        split_path = resolve_path(default_split_file, split_csv)

    silver_path = resolve_path("sinhala_silver_labels.csv", silver_csv)
    gold_path = resolve_path("annotation_with_sentiment_intent.csv", gold_csv)

    raw_df = pd.read_csv(silver_path)
    raw_df = raw_df.rename(
        columns={"Sentence": "raw_text", "intent": "intent_label", "Sentiment": "sentiment_label"}
    )
    raw_df.insert(0, "uid", np.arange(len(raw_df)))
    raw_df = raw_df[["uid", "raw_text", "sentiment_label", "intent_label"]].dropna().copy()
    raw_df["text"] = raw_df["raw_text"].apply(clean_text_runA if is_run_a else clean_text_runB)
    raw_df["sent_id"] = encode_sentiment(raw_df["sentiment_label"])
    raw_df["intent_id"] = encode_intent(raw_df["intent_label"])

    splits_df = pd.read_csv(split_path)
    splits_df["uid"] = splits_df["uid"].astype(int)

    # Index raw_df by uid to preserve exact split row ordering from splits_runA/B.csv
    raw_by_uid = raw_df.set_index("uid")

    train_uids = splits_df.loc[splits_df["split"] == "train", "uid"].values
    val_uids = splits_df.loc[splits_df["split"] == "val", "uid"].values
    test_uids = splits_df.loc[splits_df["split"] == "test", "uid"].values

    train_df = raw_by_uid.loc[train_uids].reset_index()
    val_df = raw_by_uid.loc[val_uids].reset_index()
    test_df = raw_by_uid.loc[test_uids].reset_index()

    # Merge human gold annotations onto test_df
    gold_df = pd.read_csv(gold_path, keep_default_na=False)
    gold_df["uid"] = gold_df["uid"].astype(int)
    for c in ("gold_sentiment", "gold_intent"):
        gold_df[c] = gold_df[c].astype(str).str.strip()

    test_df = test_df.merge(
        gold_df[["uid", "gold_sentiment", "gold_intent"]],
        on="uid",
        how="left",
    )
    assert test_df["gold_sentiment"].notna().all() and (test_df["gold_sentiment"] != "").all(), (
        "All 300 test rows must have valid gold_sentiment annotations"
    )
    assert test_df["gold_intent"].notna().all() and (test_df["gold_intent"] != "").all(), (
        "All 300 test rows must have valid gold_intent annotations"
    )

    test_df["silver_sent_id"] = test_df["sent_id"]
    test_df["silver_intent_id"] = test_df["intent_id"]
    test_df["gold_sent_id"] = encode_sentiment(test_df["gold_sentiment"])
    test_df["gold_intent_id"] = encode_intent(test_df["gold_intent"])

    # Integrity & leakage checks
    expected_train = 4609 if is_run_a else 4703
    expected_val = 814 if is_run_a else 831
    assert len(train_df) == expected_train, f"Expected {expected_train} train rows, got {len(train_df)}"
    assert len(val_df) == expected_val, f"Expected {expected_val} val rows, got {len(val_df)}"
    assert len(test_df) == 300, f"Expected 300 test rows, got {len(test_df)}"

    train_set, val_set, test_set = set(train_df["uid"]), set(val_df["uid"]), set(test_df["uid"])
    assert not (train_set & test_set), "Data leakage: train and test UIDs overlap!"
    assert not (val_set & test_set), "Data leakage: val and test UIDs overlap!"
    assert not (train_set & val_set), "Data leakage: train and val UIDs overlap!"
    assert not (set(train_df["text"]) & set(test_df["text"])), "Data leakage: exact duplicate text between train and test!"
    assert not (set(val_df["text"]) & set(test_df["text"])), "Data leakage: exact duplicate text between val and test!"

    return train_df, val_df, test_df


# ── Metrics Computation ───────────────────────────────────────────────────────
def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    Compute macro-F1 over classes present in y_true (matches ablation notebooks' `macro_f1`).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return float(f1[cm.sum(1) > 0].mean())


def compute_metrics(
    y_sent_true: np.ndarray,
    y_sent_pred: np.ndarray,
    y_int_true: Optional[np.ndarray] = None,
    y_int_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute all required sentiment, intent, and joint evaluation metrics.
    Supports single-task sentiment evaluation when `y_int_pred` is None.
    """
    ys_t = np.asarray(y_sent_true, dtype=np.int64)
    ys_p = np.asarray(y_sent_pred, dtype=np.int64)

    sent_acc = float((ys_t == ys_p).mean())
    sent_p, sent_r, sent_f1, sent_sup = precision_recall_fscore_support(
        ys_t, ys_p, labels=list(range(NS)), zero_division=0
    )
    sent_macro_f1_4cls = float(np.mean(sent_f1))
    sent_macro_f1_3cls = float(np.mean(sent_f1[SENT_3CLS_INDICES]))

    sent_per_class = {
        SENT_CLASSES[i]: {
            "precision": float(sent_p[i]),
            "recall": float(sent_r[i]),
            "f1": float(sent_f1[i]),
            "support": int(sent_sup[i]),
        }
        for i in range(NS)
    }

    result: Dict[str, Any] = {
        "sent_acc": sent_acc,
        "sent_f1_4cls": sent_macro_f1_4cls,
        "sent_f1_3cls": sent_macro_f1_3cls,
        "intent_acc": None,
        "intent_f1": None,
        "joint_acc": None,
        "sentiment_per_class": sent_per_class,
        "intent_per_class": None,
        "intent_confusion_matrix": None,
    }

    if y_int_true is not None and y_int_pred is not None:
        yi_t = np.asarray(y_int_true, dtype=np.int64)
        yi_p = np.asarray(y_int_pred, dtype=np.int64)

        intent_acc = float((yi_t == yi_p).mean())
        int_p, int_r, int_f1, int_sup = precision_recall_fscore_support(
            yi_t, yi_p, labels=list(range(NI)), zero_division=0
        )
        intent_macro_f1 = float(np.mean(int_f1))
        joint_acc = float(((ys_t == ys_p) & (yi_t == yi_p)).mean())

        cm_int = confusion_matrix(yi_t, yi_p, labels=list(range(NI)))

        intent_per_class = {
            INTENT_CLASSES[i]: {
                "precision": float(int_p[i]),
                "recall": float(int_r[i]),
                "f1": float(int_f1[i]),
                "support": int(int_sup[i]),
            }
            for i in range(NI)
        }

        result.update({
            "intent_acc": intent_acc,
            "intent_f1": intent_macro_f1,
            "joint_acc": joint_acc,
            "intent_per_class": intent_per_class,
            "intent_confusion_matrix": cm_int.tolist(),
        })

    return result


def save_intent_confusion_matrix(
    cm: np.ndarray,
    csv_path: str,
    png_path: str,
    title: str = "Intent Confusion Matrix (Gold Test)",
) -> None:
    """Save the 6x6 intent confusion matrix as both CSV and PNG."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in INTENT_CLASSES],
        columns=[f"pred_{c}" for c in INTENT_CLASSES],
    )
    cm_df.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(NI),
        yticks=np.arange(NI),
        xticklabels=INTENT_CLASSES,
        yticklabels=INTENT_CLASSES,
        ylabel="Gold Intent",
        xlabel="Predicted Intent",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(NI):
        for j in range(NI):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_save(
    notebook_name: str,
    model_name: str,
    run: str,
    seed: Any,
    test_df: pd.DataFrame,
    pred_sent_id: np.ndarray,
    pred_intent_id: Optional[np.ndarray] = None,
    sent_probs: Optional[np.ndarray] = None,
    intent_probs: Optional[np.ndarray] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    results_dir: str = "results",
    preds_dir: str = "preds",
) -> Dict[str, Any]:
    """
    Evaluate model predictions on the 300 gold test rows and save:
    1. `results/{notebook_name}_{run}_{seed}.json`
    2. `preds/{model_name}_{run}_{seed}.csv`
    3. `results/cm_intent_{notebook_name}_{run}_{seed}.csv` and `.png` (if intent predictions exist)
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    y_sent_gold = test_df["gold_sent_id"].values
    y_int_gold = test_df["gold_intent_id"].values

    metrics = compute_metrics(
        y_sent_true=y_sent_gold,
        y_sent_pred=pred_sent_id,
        y_int_true=y_int_gold if pred_intent_id is not None else None,
        y_int_pred=pred_intent_id,
    )

    payload = {
        "notebook": notebook_name,
        "model": model_name,
        "run": run,
        "seed": seed,
        "n_test": len(test_df),
        **metrics,
    }
    if extra_meta:
        payload["meta"] = extra_meta

    # Save JSON
    json_path = os.path.join(results_dir, f"{notebook_name}_{run}_{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save per-row predictions CSV
    pred_df = pd.DataFrame({
        "uid": test_df["uid"].values,
        "raw_text": test_df["raw_text"].values,
        "text": test_df["text"].values,
        "gold_sent_id": y_sent_gold,
        "gold_intent_id": y_int_gold,
        "gold_sentiment": [ID_TO_SENT[int(i)] for i in y_sent_gold],
        "gold_intent": [ID_TO_INTENT[int(i)] for i in y_int_gold],
        "pred_sent_id": np.asarray(pred_sent_id, dtype=np.int64),
        "pred_sentiment": [ID_TO_SENT[int(i)] for i in pred_sent_id],
    })
    if pred_intent_id is not None:
        pred_df["pred_intent_id"] = np.asarray(pred_intent_id, dtype=np.int64)
        pred_df["pred_intent"] = [ID_TO_INTENT[int(i)] for i in pred_intent_id]
    else:
        pred_df["pred_intent_id"] = -1
        pred_df["pred_intent"] = ""

    if sent_probs is not None:
        for idx, cname in enumerate(SENT_CLASSES):
            pred_df[f"prob_sent_{cname}"] = np.round(sent_probs[:, idx], 6)
    if intent_probs is not None:
        for idx, cname in enumerate(INTENT_CLASSES):
            pred_df[f"prob_intent_{cname}"] = np.round(intent_probs[:, idx], 6)

    pred_csv_path = os.path.join(preds_dir, f"{model_name}_{run}_{seed}.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    # Save confusion matrix for intent if available
    if metrics["intent_confusion_matrix"] is not None:
        cm_arr = np.array(metrics["intent_confusion_matrix"], dtype=np.int64)
        cm_csv = os.path.join(results_dir, f"cm_intent_{notebook_name}_{run}_{seed}.csv")
        cm_png = os.path.join(results_dir, f"cm_intent_{notebook_name}_{run}_{seed}.png")
        save_intent_confusion_matrix(
            cm_arr,
            csv_path=cm_csv,
            png_path=cm_png,
            title=f"Intent CM: {model_name} ({run}, seed={seed})",
        )

    return payload


def summarize_Across_seeds(run_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and std over seeds for a list of run metric dicts."""
    keys = ["sent_acc", "sent_f1_3cls", "sent_f1_4cls", "intent_acc", "intent_f1", "joint_acc"]
    summary: Dict[str, Any] = {
        "model": run_payloads[0]["model"],
        "notebook": run_payloads[0]["notebook"],
        "run": run_payloads[0]["run"],
        "n_seeds": len(run_payloads),
        "seeds": [p["seed"] for p in run_payloads],
    }
    for k in keys:
        vals = [p[k] for p in run_payloads if p.get(k) is not None]
        if not vals:
            summary[f"{k}_mean"] = None
            summary[f"{k}_std"] = None
            summary[f"{k}_str"] = "—"
        elif len(vals) == 1:
            summary[f"{k}_mean"] = float(vals[0])
            summary[f"{k}_std"] = 0.0
            summary[f"{k}_str"] = f"{vals[0]:.3f}"
        else:
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=0))
            summary[f"{k}_mean"] = m
            summary[f"{k}_std"] = s
            summary[f"{k}_str"] = f"{m:.3f} ± {s:.3f}"
    return summary

