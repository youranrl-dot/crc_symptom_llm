"""
Step 7: Predictive Validity of Symptom Clusters
- Logistic regression: per-cluster symptom burden → clinical outcomes
- Outcomes: in-hospital mortality, 30-day readmission, 1-year mortality
- Covariates: age, sex
- 3 clusters entered simultaneously as continuous predictors

Requirements:
    pip install statsmodels pandas numpy

Usage:
    python 07_predictive_validity.py --base_dir .
"""

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Cluster membership (Louvain, Claude Haiku predictions, phi>=0.10)
CLUSTER_MAP = {
    # Cluster 1: Constitutional / Systemic
    "pred_lack_of_energy": 1, "pred_pain": 1, "pred_shortness_of_breath": 1,
    "pred_problems_with_urination": 1, "pred_feeling_drowsy": 1,
    "pred_difficulty_sleeping": 1, "pred_feeling_sad": 1,
    "pred_worrying": 1, "pred_feeling_nervous": 1,
    # Cluster 2: CRC Disease-Specific
    "pred_diarrhoea": 2, "pred_constipation": 2, "pred_blood_in_stool": 2,
    "pred_abdominal_pain": 2, "pred_nausea": 2, "pred_weight_loss": 2,
    "pred_lack_of_appetite": 2,
    # Cluster 3: Gastrointestinal
    "pred_flatulence_gas": 3, "pred_frequent_bowel_movements": 3,
    "pred_feeling_bloated": 3, "pred_vomiting": 3, "pred_dizziness": 3,
}

CLUSTER_NAMES = {1: "C1-Systemic", 2: "C2-CRC Disease-Specific", 3: "C3-Gastrointestinal"}


def load_symptom_burden(pred_file: str, prevalence_threshold: float = 0.05):
    """Patient-level symptom burden scores per cluster."""
    df = pd.read_csv(pred_file)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    pt = df.groupby("subject_id")[pred_cols].max().reset_index()
    prev = pt[pred_cols].mean()
    included = set(prev[prev >= prevalence_threshold].index.tolist())

    for cluster_id in [1, 2, 3]:
        syms = [s for s, c in CLUSTER_MAP.items() if c == cluster_id and s in included]
        pt[f"burden_c{cluster_id}"] = pt[syms].sum(axis=1)
        print(f"  Cluster {cluster_id}: {len(syms)} symptoms included")

    return pt[["subject_id", "burden_c1", "burden_c2", "burden_c3"]]


def build_outcomes(adm_file: str, pts_file: str, subject_ids):
    """Construct clinical outcome variables."""
    adm = pd.read_csv(adm_file)
    pts = pd.read_csv(pts_file)
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])

    crc_adm = adm[adm["subject_id"].isin(subject_ids)].sort_values(
        ["subject_id", "admittime"]
    )

    # In-hospital mortality: died during ANY CRC admission
    died = (
        crc_adm[crc_adm["hospital_expire_flag"] == 1][["subject_id"]]
        .drop_duplicates()
        .assign(inhosp_death=1)
    )

    # 30-day readmission: readmitted within 30 days of first admission discharge
    first_adm = crc_adm.groupby("subject_id").first().reset_index()
    future = crc_adm[["subject_id", "admittime"]].merge(
        first_adm[["subject_id", "dischtime"]], on="subject_id"
    )
    future["days"] = (future["admittime"] - future["dischtime"]).dt.days
    readmit30 = (
        future[(future["days"] > 0) & (future["days"] <= 30)][["subject_id"]]
        .drop_duplicates()
        .assign(readmit_30d=1)
    )

    # 1-year mortality: died within 365 days of first admission date
    pts2 = pts[["subject_id", "dod", "anchor_age", "gender"]].copy()
    pts2["dod"] = pd.to_datetime(pts2["dod"], errors="coerce")
    pts2 = pts2.merge(first_adm[["subject_id", "admittime"]], on="subject_id", how="left")
    pts2["days_to_death"] = (pts2["dod"] - pts2["admittime"]).dt.days
    pts2["mort_1yr"] = (
        (pts2["days_to_death"] >= 0) & (pts2["days_to_death"] <= 365)
    ).astype(int)
    pts2["gender_m"] = (pts2["gender"] == "M").astype(int)

    return died, readmit30, pts2


def run_logistic(df, outcome_col, burden_cols, covariates):
    """Fit logistic regression; return OR table."""
    sub = df[[outcome_col] + burden_cols + covariates].dropna()
    X = sm.add_constant(sub[burden_cols + covariates])
    model = sm.Logit(sub[outcome_col], X).fit(disp=0)

    rows = []
    for bc in burden_cols:
        OR = np.exp(model.params[bc])
        ci = np.exp(model.conf_int().loc[bc])
        p = model.pvalues[bc]
        rows.append({
            "Cluster": CLUSTER_NAMES.get(int(bc[-1]), bc),
            "OR": round(OR, 3),
            "CI_lower": round(ci[0], 3),
            "CI_upper": round(ci[1], 3),
            "p_value": round(p, 4),
            "sig": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")),
        })
    return pd.DataFrame(rows), sub[outcome_col].sum()


def main(base_dir: str = "."):
    print(f"\n{'='*55}")
    print("Predictive Validity Analysis")
    print(f"{'='*55}")

    pred_file = str(Path(base_dir) / "files/Final/predictions_claude_syn_full .csv")
    adm_file  = str(Path(base_dir) / "files/admissions.csv")
    pts_file  = str(Path(base_dir) / "files/patients.csv")

    # Load burden scores
    print("\nLoading symptom burden scores (Claude Haiku)...")
    burden = load_symptom_burden(pred_file)
    subject_ids = burden["subject_id"].values

    # Build outcomes
    print("\nBuilding outcome variables...")
    died, readmit30, pts2 = build_outcomes(adm_file, pts_file, subject_ids)

    # Merge
    df = burden.merge(died, on="subject_id", how="left")
    df["inhosp_death"] = df["inhosp_death"].fillna(0).astype(int)
    df = df.merge(readmit30, on="subject_id", how="left")
    df["readmit_30d"] = df["readmit_30d"].fillna(0).astype(int)
    df = df.merge(pts2[["subject_id", "mort_1yr", "anchor_age", "gender_m"]],
                  on="subject_id", how="left")

    N = len(df)
    print(f"\nN = {N} patients")
    print(f"  In-hospital mortality: n={df['inhosp_death'].sum()} ({df['inhosp_death'].mean()*100:.1f}%)")
    print(f"  30-day readmission:    n={df['readmit_30d'].sum()} ({df['readmit_30d'].mean()*100:.1f}%)")
    print(f"  1-year mortality:      n={df['mort_1yr'].sum()} ({df['mort_1yr'].mean()*100:.1f}%)")

    burden_cols = ["burden_c1", "burden_c2", "burden_c3"]
    covariates  = ["anchor_age", "gender_m"]

    outcomes = [
        ("In-hospital mortality", "inhosp_death"),
        ("30-day readmission",    "readmit_30d"),
        ("1-year mortality",      "mort_1yr"),
    ]

    all_results = []
    for out_name, out_col in outcomes:
        result_df, n_events = run_logistic(df, out_col, burden_cols, covariates)
        result_df.insert(0, "Outcome", out_name)
        result_df.insert(1, "N_events", n_events)
        all_results.append(result_df)
        print(f"\n{out_name} (n={n_events}):")
        for _, row in result_df.iterrows():
            print(f"  {row['Cluster']}: OR={row['OR']} "
                  f"(95%CI {row['CI_lower']}–{row['CI_upper']}), "
                  f"p={row['p_value']} {row['sig']}")

    # Save results
    out_csv = str(Path(base_dir) / "predictive_validity_results.csv")
    pd.concat(all_results, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".")
    args = parser.parse_args()
    main(base_dir=args.base_dir)
