"""
Step 4: Evaluation — precision, recall, F1 per symptom + macro/micro
Ground truth: annotation_sample_1000_cchpi_annotated.csv
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/pipeline/data"
ANN_FILE = "/sessions/stoic-practical-fermi/mnt/MIMIC_LLM/files/annotation_sample_1000_cchpi_annotated.csv"
OUT_DIR  = DATA_DIR

# Symptoms with 0 positives → excluded from evaluation
EXCLUDE_ZERO = True

def evaluate(gt_df, pred_file, method_name):
    pred_df = pd.read_csv(pred_file)
    # align by note_id
    merged = gt_df.merge(pred_df, on="note_id", how="inner", suffixes=("_gt", "_pred"))
    print(f"\n  [{method_name}] Matched notes: {len(merged)}")

    symptoms = [c.replace("label_", "") for c in gt_df.columns if c.startswith("label_")]
    rows = []
    for sym in symptoms:
        gt_col   = f"label_{sym}"
        pred_col = f"pred_{sym}"
        if gt_col not in merged.columns or pred_col not in merged.columns:
            continue
        y_true = merged[gt_col].fillna(0).astype(int)
        y_pred = merged[pred_col].fillna(0).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else float("nan"))

        rows.append({
            "symptom":    sym,
            "n_pos_gt":   int(y_true.sum()),
            "n_pos_pred": int(y_pred.sum()),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision":  round(precision, 4) if not np.isnan(precision) else None,
            "recall":     round(recall,    4) if not np.isnan(recall)    else None,
            "f1":         round(f1,        4) if not np.isnan(f1)        else None,
            "method":     method_name
        })

    result = pd.DataFrame(rows)

    # Exclude zero-positive symptoms for aggregate metrics
    eval_df = result[result["n_pos_gt"] > 0] if EXCLUDE_ZERO else result

    # Macro (unweighted mean of per-symptom metrics)
    macro_p  = eval_df["precision"].dropna().mean()
    macro_r  = eval_df["recall"].dropna().mean()
    macro_f1 = eval_df["f1"].dropna().mean()

    # Micro (sum of TP/FP/FN)
    tp_sum = eval_df["TP"].sum()
    fp_sum = eval_df["FP"].sum()
    fn_sum = eval_df["FN"].sum()
    micro_p  = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    micro_r  = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0)

    print(f"  Macro P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")
    print(f"  Micro P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f1:.4f}")
    print(f"  (Evaluated on {len(eval_df)} symptoms with n_pos > 0)")

    return result, {
        "method":   method_name,
        "macro_P":  round(macro_p,  4),
        "macro_R":  round(macro_r,  4),
        "macro_F1": round(macro_f1, 4),
        "micro_P":  round(micro_p,  4),
        "micro_R":  round(micro_r,  4),
        "micro_F1": round(micro_f1, 4),
        "n_symptoms_eval": len(eval_df)
    }


if __name__ == "__main__":
    print("Loading ground truth ...")
    gt_df = pd.read_csv(ANN_FILE)
    label_cols = [c for c in gt_df.columns if c.startswith("label_")]
    print(f"  GT rows: {len(gt_df)}  |  Symptoms: {len(label_cols)}")

    methods = [
        ("Rule-based",  f"{DATA_DIR}/predictions_rulebased_gt1000.csv"),
        ("NER",         f"{DATA_DIR}/predictions_ner_gt1000.csv"),
    ]

    all_per_symptom = []
    summary_rows    = []

    for name, path in methods:
        if not os.path.exists(path):
            print(f"\n  SKIP {name}: file not found ({path})")
            continue
        per_sym, summary = evaluate(gt_df, path, name)
        all_per_symptom.append(per_sym)
        summary_rows.append(summary)

    if not summary_rows:
        print("\nNo results to save. Run Steps 2 and 3 first.")
    else:
        # Combined per-symptom table
        combined = pd.concat(all_per_symptom, ignore_index=True)
        combined.to_csv(f"{OUT_DIR}/eval_per_symptom.csv", index=False)

        # Pivot: symptom × method
        pivot = combined.pivot_table(
            index="symptom",
            columns="method",
            values=["precision", "recall", "f1", "n_pos_gt"]
        ).round(4)
        pivot.to_csv(f"{OUT_DIR}/eval_pivot.csv")

        # Summary
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f"{OUT_DIR}/eval_summary.csv", index=False)

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))
        print(f"\nSaved:")
        print(f"  {OUT_DIR}/eval_per_symptom.csv")
        print(f"  {OUT_DIR}/eval_pivot.csv")
        print(f"  {OUT_DIR}/eval_summary.csv")
