"""Inter-rater reliability on the 200-note gold standard (Section 3.2, Table S2).

The workbook produced during adjudication already stores per-symptom kappa and
percent agreement computed on the independent, pre-adjudication labels. This
module reads that workbook, recomputes the pooled and macro summaries from the
per-symptom values, and joins the adjudicated gold-standard prevalence so the
supplementary table is self-contained.
"""

from __future__ import annotations

import pandas as pd

from .config import Paths

__all__ = ["load_kappa_results", "summarise_irr", "build_table_s2"]

#: Landis-Koch boundaries used for the category column.
LANDIS_KOCH = [
    (0.80, "almost perfect"),
    (0.60, "substantial"),
    (0.40, "moderate"),
    (0.20, "fair"),
    (0.00, "slight"),
]


def _landis_koch(kappa: float) -> str:
    if pd.isna(kappa):
        return "undefined"
    if kappa < 0:
        return "poor"
    for cutoff, label in LANDIS_KOCH:
        if kappa >= cutoff:
            return label
    return "slight"


def load_kappa_results(paths: Paths | None = None) -> pd.DataFrame:
    """Read the ``Kappa_Results`` and ``Final_Consensus_200`` sheets and join them."""
    paths = paths or Paths()
    book = pd.ExcelFile(paths.kappa_workbook)
    kappa = book.parse("Kappa_Results")
    gold = book.parse("Final_Consensus_200")

    label_cols = [c for c in gold.columns if c.startswith("label_")]
    gold_counts = {c.removeprefix("label_"): int(gold[c].sum()) for c in label_cols}
    n_notes = len(gold)

    kappa = kappa.copy()
    kappa["symptom_key"] = kappa["symptom"].str.strip().str.replace(" ", "_")
    kappa["gold_n"] = kappa["symptom_key"].map(gold_counts)
    if kappa["gold_n"].isna().any():
        unmatched = kappa.loc[kappa["gold_n"].isna(), "symptom"].tolist()
        raise ValueError(f"symptoms absent from the consensus sheet: {unmatched}")
    kappa["gold_prevalence_pct"] = (kappa["gold_n"] / n_notes * 100).round(1)
    kappa["landis_koch"] = kappa["cohens_kappa"].map(_landis_koch)
    return kappa


def summarise_irr(kappa: pd.DataFrame, n_notes: int = 200) -> dict:
    """Pooled and macro summaries reported in Section 3.2.

    ``pct_agreement`` in the workbook is a proportion per symptom; the pooled
    (micro) agreement is its mean across symptoms, which equals the proportion of
    concordant symptom-note pairs because every symptom is scored on every note.
    """
    n_pairs = n_notes * len(kappa)
    pooled_agreement = float(kappa["pct_agreement"].mean())
    discordant = int(round((1 - pooled_agreement) * n_pairs))
    estimable = kappa["cohens_kappa"].dropna()
    return {
        "n_notes": n_notes,
        "n_symptoms": len(kappa),
        "n_symptom_note_pairs": n_pairs,
        "n_discordant": discordant,
        "pct_discordant": round(discordant / n_pairs * 100, 1),
        "pooled_percent_agreement": round(pooled_agreement * 100, 1),
        "macro_kappa": round(float(estimable.mean()), 4),
        "n_estimable": int(estimable.size),
        "n_not_estimable": int(kappa["cohens_kappa"].isna().sum()),
        "n_kappa_ge_060": int((estimable >= 0.60).sum()),
        "n_kappa_lt_020": int((estimable < 0.20).sum()),
    }


def build_table_s2(paths: Paths | None = None) -> pd.DataFrame:
    """Supplementary Table S2 — per-symptom inter-rater reliability."""
    kappa = load_kappa_results(paths)
    table = (
        kappa.assign(
            symptom=lambda d: d["symptom"].str.strip().str.capitalize(),
            percent_agreement=lambda d: (d["pct_agreement"] * 100).round(1),
            cohens_kappa=lambda d: d["cohens_kappa"].round(3),
        )
        .loc[
            :,
            [
                "symptom",
                "kim_pos",
                "yr_pos",
                "gold_n",
                "gold_prevalence_pct",
                "percent_agreement",
                "cohens_kappa",
                "landis_koch",
            ],
        ]
        .rename(
            columns={
                "kim_pos": "annotator_1_positive",
                "yr_pos": "annotator_2_positive",
                "gold_n": "gold_standard_n",
                "gold_prevalence_pct": "gold_standard_pct",
            }
        )
        .sort_values("cohens_kappa", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    if paths is not None:
        table.to_csv(paths.ensure_results() / "table_s2_inter_rater.csv", index=False)
    return table
