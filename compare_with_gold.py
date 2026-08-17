#!/usr/bin/env python3
"""Benchmark symptom-extraction output against the adjudicated 200-note gold standard.

Produces Table 2 (Macro/Micro F1, Macro precision/recall), Table 3 (pooled
confusion matrix), and the per-symptom F1 scores behind Figure 1 — for one or
many prediction files in a single pass.

    python compare_with_gold.py \
        --pred rule_based=data/preds_rule_based.csv \
        --pred claude=data/predictions_claude_haiku_full.csv \
        --pred gemini=data/preds_gemini.csv \
        --gold data/200_Note_Adjudication_Kappa_Results.xlsx \
        --out results/benchmark

Label contract
--------------
``symptoms.json`` holds the canonical 46 keys. The gold standard carries them as
``label_<key>`` columns; prediction files carry them as ``pred_<key>`` (bare
``<key>`` is also accepted). Any key missing from either side is a hard error —
silently scoring a subset is how a benchmark quietly stops meaning anything.

Missing predictions
-------------------
Notes whose CC/HPI section could not be extracted appear in some prediction files
with all-null labels (``error=missing_cc_hpi``). ``--na-policy`` decides what
happens to them; the default refuses to guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
GOLD_SHEET = "Final_Consensus_200"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_labels(path: Path | None = None) -> list[str]:
    """Canonical 46-symptom key list."""
    path = path or HERE / "symptoms.json"
    with open(path, encoding="utf-8") as handle:
        labels = json.load(handle)["labels"]
    placeholders = [k for k in labels if k.upper().startswith("TODO")]
    if placeholders:
        raise ValueError(
            f"{path} still contains placeholders: {placeholders}. "
            "Fill the schema before benchmarking."
        )
    return labels


def load_gold(path: Path, labels: list[str]) -> pd.DataFrame:
    """Adjudicated consensus labels, one row per note, columns = canonical keys."""
    gold = (
        pd.read_excel(path, sheet_name=GOLD_SHEET)
        if path.suffix.lower() in {".xlsx", ".xlsm"}
        else pd.read_csv(path)
    )
    rename = {f"label_{k}": k for k in labels if f"label_{k}" in gold.columns}
    gold = gold.rename(columns=rename)
    missing = [k for k in labels if k not in gold.columns]
    if missing:
        raise KeyError(f"gold standard is missing {len(missing)} labels: {missing[:5]}")
    return gold[["note_id", *labels]]


def load_predictions(path: Path, labels: list[str]) -> pd.DataFrame:
    """Prediction file, normalised to ``note_id`` + canonical keys."""
    preds = pd.read_csv(path)
    rename = {f"pred_{k}": k for k in labels if f"pred_{k}" in preds.columns}
    preds = preds.rename(columns=rename)
    missing = [k for k in labels if k not in preds.columns]
    if missing:
        raise KeyError(
            f"{path.name} is missing {len(missing)} of the 46 labels: {missing[:5]}"
        )
    return preds[["note_id", *labels]]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def align(
    gold: pd.DataFrame, preds: pd.DataFrame, labels: list[str], na_policy: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inner-join on ``note_id`` and return aligned (y_true, y_pred) matrices."""
    merged = gold.merge(preds, on="note_id", how="inner", suffixes=("_gold", "_pred"))
    gold_cols = [f"{k}_gold" for k in labels]
    pred_cols = [f"{k}_pred" for k in labels]

    n_matched = len(merged)
    n_missing_notes = len(gold) - n_matched
    n_null_cells = int(merged[pred_cols].isna().sum().sum())
    n_null_notes = int(merged[pred_cols].isna().any(axis=1).sum())

    if n_null_cells:
        if na_policy == "error":
            raise ValueError(
                f"{n_null_cells} null predictions across {n_null_notes} note(s). "
                "Re-run those notes, or choose --na-policy drop|zero."
            )
        if na_policy == "drop":
            merged = merged[~merged[pred_cols].isna().any(axis=1)]
        elif na_policy == "zero":
            merged[pred_cols] = merged[pred_cols].fillna(0)

    y_true = merged[gold_cols].to_numpy(dtype=int)
    y_pred = merged[pred_cols].to_numpy(dtype=int)
    info = {
        "n_gold_notes": len(gold),
        "n_scored_notes": len(merged),
        "n_unmatched_notes": n_missing_notes,
        "n_null_prediction_notes": n_null_notes,
        "n_null_prediction_cells": n_null_cells,
        "na_policy": na_policy,
    }
    return y_true, y_pred, info


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def per_symptom_scores(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """One row per symptom: confusion counts, precision/recall/F1, kappa, support."""
    rows = []
    for j, label in enumerate(labels):
        t, p = y_true[:, j], y_pred[:, j]
        tp = int(((t == 1) & (p == 1)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        fn = int(((t == 1) & (p == 0)).sum())
        tn = int(((t == 0) & (p == 0)).sum())
        precision, recall, f1 = _prf(tp, fp, fn)
        # kappa is undefined when both raters are constant and identical
        kappa = (
            float(cohen_kappa_score(t, p))
            if len(set(t.tolist())) > 1 or len(set(p.tolist())) > 1
            else np.nan
        )
        rows.append(
            {
                "symptom": label,
                "support_gold": int(t.sum()),
                "predicted_positive": int(p.sum()),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "cohens_kappa": None if np.isnan(kappa) else round(kappa, 4),
            }
        )
    return pd.DataFrame(rows)


def overall_scores(per_symptom: pd.DataFrame) -> dict:
    """Table 2 + Table 3 for one method.

    Macro metrics average the per-symptom values with equal weight (symptoms with
    no gold positives contribute 0, matching the manuscript). Micro F1 pools the
    confusion counts across all symptoms first.
    """
    tp = int(per_symptom["TP"].sum())
    fp = int(per_symptom["FP"].sum())
    fn = int(per_symptom["FN"].sum())
    tn = int(per_symptom["TN"].sum())
    _, _, micro_f1 = _prf(tp, fp, fn)
    return {
        "macro_f1": round(float(per_symptom["f1"].mean()), 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(float(per_symptom["precision"].mean()), 4),
        "macro_recall": round(float(per_symptom["recall"].mean()), 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_pred_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, _, raw = value.partition("=")
    return name, Path(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pred", action="append", required=True, metavar="NAME=PATH",
        help="prediction file; repeat for each method",
    )
    parser.add_argument(
        "--gold", type=Path,
        default=HERE / "data" / "200_Note_Adjudication_Kappa_Results.xlsx",
    )
    parser.add_argument("--symptoms", type=Path, default=HERE / "symptoms.json")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "benchmark")
    parser.add_argument(
        "--na-policy", choices=["error", "drop", "zero"], default="error",
        help="how to treat notes with null predictions (default: refuse)",
    )
    args = parser.parse_args(argv)

    labels = load_labels(args.symptoms)
    gold = load_gold(args.gold, labels)
    print(f"gold standard: {len(gold)} notes x {len(labels)} symptoms "
          f"= {len(gold) * len(labels):,} symptom-note pairs\n")

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows, f1_columns = [], {}

    for spec in args.pred:
        name, path = _parse_pred_arg(spec)
        preds = load_predictions(path, labels)
        y_true, y_pred, info = align(gold, preds, labels, args.na_policy)

        per_symptom = per_symptom_scores(y_true, y_pred, labels)
        per_symptom.to_csv(args.out / f"per_symptom_{name}.csv", index=False)
        f1_columns[name] = per_symptom.set_index("symptom")["f1"]

        overall = overall_scores(per_symptom)
        summary_rows.append({"method": name, **info, **overall})

        print(f"[{name}]  {path.name}")
        print(f"   scored {info['n_scored_notes']}/{info['n_gold_notes']} notes"
              + (f"   ({info['n_null_prediction_notes']} note(s) with null "
                 f"predictions, policy={info['na_policy']})"
                 if info["n_null_prediction_cells"] else ""))
        print(f"   Macro F1 {overall['macro_f1']:.4f}   Micro F1 {overall['micro_f1']:.4f}"
              f"   Macro P {overall['macro_precision']:.4f}"
              f"   Macro R {overall['macro_recall']:.4f}")
        print(f"   TP {overall['TP']}  FP {overall['FP']}  "
              f"FN {overall['FN']}  TN {overall['TN']}\n")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "table2_table3_summary.csv", index=False)
    pd.DataFrame(f1_columns).to_csv(args.out / "figure1_per_symptom_f1.csv")

    if len(summary) > 1 and summary["n_scored_notes"].nunique() > 1:
        print("WARNING: methods were scored on different numbers of notes; "
              "the columns are not directly comparable.\n")

    print(summary[["method", "n_scored_notes", "macro_f1", "micro_f1",
                   "macro_precision", "macro_recall", "TP", "FP", "FN", "TN"]]
          .to_string(index=False))
    print(f"\nwrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
