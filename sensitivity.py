"""The four pre-specified network sensitivity analyses, plus an outcome-definition grid.

Section 3.6 of the manuscript reports:
  1. phi-threshold sweep (0.10 / 0.15 / 0.20)
  2. Louvain reproducibility over 100 random seeds
  3. first-note vs any-note symptom aggregation
  4. bootstrap resampling of patients (200 iterations)

``outcome_definition_grid`` is an additional diagnostic: it shows how much each
outcome count moves when the admission set or the anchor is changed. It exists
because two different definitions were originally in circulation for the same
outcome names, and the grid makes that class of error visible immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .config import (
    N_BOOTSTRAP,
    N_LOUVAIN_SEEDS,
    PHI_THRESHOLD,
    PHI_THRESHOLD_GRID,
    PREVALENCE_THRESHOLD,
    RANDOM_STATE,
    Paths,
    pretty_symptom,
)
from .data import Cohort, patient_level_matrix
from .network import SymptomNetwork, build_network

__all__ = [
    "threshold_sensitivity",
    "louvain_seed_stability",
    "bootstrap_stability",
    "first_note_vs_any_note",
    "outcome_definition_grid",
]


def threshold_sensitivity(
    matrix: pd.DataFrame, reference: SymptomNetwork, paths: Paths | None = None
) -> pd.DataFrame:
    """Re-run clustering at each phi threshold and compare against the reference.

    Reports ARI against the reference partition as well as cluster count, because
    "three clusters at both thresholds" can be true while membership changes
    substantially — which is exactly what happens between 0.10 and 0.15 here.
    """
    ref_labels = reference.labels_for(reference.retained)
    rows = []
    for threshold in PHI_THRESHOLD_GRID:
        net = build_network(
            matrix,
            phi_threshold=threshold,
            prevalence_threshold=reference.prevalence_threshold,
            random_state=reference.random_state,
        )
        rows.append(
            {
                "phi_threshold": threshold,
                "n_edges": net.n_edges,
                "n_clusters": net.n_communities,
                "cluster_sizes": "/".join(str(s) for s in net.community_sizes),
                "ari_vs_reference": round(
                    adjusted_rand_score(ref_labels, net.labels_for(reference.retained)), 3
                ),
                "singletons": ", ".join(
                    pretty_symptom(s) for s in sorted(net.singletons())
                ),
            }
        )
    table = pd.DataFrame(rows)
    if paths is not None:
        table.to_csv(paths.ensure_results() / "sensitivity_phi_threshold.csv", index=False)
    return table


def louvain_seed_stability(
    reference: SymptomNetwork, n_seeds: int = N_LOUVAIN_SEEDS
) -> dict:
    """Re-run Louvain across ``n_seeds`` seeds on the fixed reference graph."""
    import community as community_louvain

    ref_labels = reference.labels_for(reference.retained)
    cluster_counts, aris = [], []
    for seed in range(n_seeds):
        part = community_louvain.best_partition(
            reference.graph, weight="weight", random_state=seed
        )
        cluster_counts.append(len(set(part.values())))
        aris.append(
            adjusted_rand_score(ref_labels, [part[s] for s in reference.retained])
        )
    aris = np.asarray(aris)
    return {
        "n_seeds": n_seeds,
        "n_three_cluster": int(sum(c == 3 for c in cluster_counts)),
        "ari_mean": round(float(aris.mean()), 3),
        "ari_min": round(float(aris.min()), 3),
        "ari_max": round(float(aris.max()), 3),
    }


def bootstrap_stability(
    matrix: pd.DataFrame,
    reference: SymptomNetwork,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Resample patients with replacement and re-derive the network each time.

    ARI is computed over the symptoms common to both partitions, because a
    resample can push a borderline symptom below the prevalence threshold.
    """
    rng = np.random.default_rng(random_state)
    n = len(matrix)
    cluster_counts, aris = [], []

    for _ in range(n_boot):
        resample = matrix.iloc[rng.choice(n, n, replace=True)]
        net = build_network(
            resample,
            phi_threshold=reference.phi_threshold,
            prevalence_threshold=reference.prevalence_threshold,
            random_state=reference.random_state,
        )
        common = [s for s in reference.retained if s in net.partition]
        cluster_counts.append(net.n_communities)
        aris.append(
            adjusted_rand_score(
                reference.labels_for(common), [net.partition[s] for s in common]
            )
        )

    aris = np.asarray(aris)
    n_three = int(sum(c == 3 for c in cluster_counts))
    return {
        "n_bootstrap": n_boot,
        "n_three_cluster": n_three,
        "pct_three_cluster": round(n_three / n_boot * 100, 1),
        "ari_mean": round(float(aris.mean()), 3),
        "ari_sd": round(float(aris.std(ddof=1)), 3),
        "ari_median": round(float(np.median(aris)), 3),
        "ari_min": round(float(aris.min()), 3),
        "ari_max": round(float(aris.max()), 3),
    }


def first_note_vs_any_note(
    preds: pd.DataFrame,
    subject_ids: list[int],
    prevalence_threshold: float = PREVALENCE_THRESHOLD,
    paths: Paths | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Compare any-note aggregation with first-note-only aggregation.

    Returns a summary dict and the table of symptoms that cross the prevalence
    threshold when aggregation is restricted to the first note.
    """
    any_note = patient_level_matrix(preds, subject_ids)
    first_note = patient_level_matrix(preds, subject_ids, first_note_only=True)

    summary = {
        "mean_symptoms_any_note": round(float(any_note.sum(axis=1).mean()), 2),
        "mean_symptoms_first_note": round(float(first_note.sum(axis=1).mean()), 2),
    }

    rows = []
    for col in any_note.columns:
        a, f = any_note[col].mean(), first_note[col].mean()
        if a >= prevalence_threshold and f < prevalence_threshold:
            rows.append(
                {
                    "symptom": pretty_symptom(col),
                    "any_note_pct": round(a * 100, 1),
                    "first_note_pct": round(f * 100, 1),
                }
            )
    dropped = pd.DataFrame(rows).sort_values("any_note_pct", ascending=False)

    if paths is not None:
        dropped.to_csv(
            paths.ensure_results() / "sensitivity_first_note_dropouts.csv", index=False
        )
    return summary, dropped


def outcome_definition_grid(cohort: Cohort, paths: Paths | None = None) -> pd.DataFrame:
    """Outcome counts under alternative admission sets and anchors.

    The published definitions are the ``all admissions`` rows. This grid is what
    surfaces a mismatch if a downstream model is ever fitted against a different
    definition than the one Table 1 reports.
    """
    note_hadms = set(cohort.notes["hadm_id"])
    admission_sets = {
        "all admissions": cohort.admissions,
        "CRC-coded admissions": cohort.admissions[
            cohort.admissions["hadm_id"].isin(cohort.crc_hadm_ids)
        ],
        "note-bearing admissions": cohort.admissions[
            cohort.admissions["hadm_id"].isin(note_hadms)
        ],
    }
    dod = cohort.date_of_death
    n = cohort.n
    rows = []

    for name, adm in admission_sets.items():
        adm = adm.sort_values(["subject_id", "admittime"])
        first = adm.groupby("subject_id").first()

        in_hosp = int(
            adm.groupby("subject_id")["hospital_expire_flag"]
            .max()
            .reindex(cohort.subject_ids)
            .fillna(0)
            .sum()
        )

        for anchor_name, anchor in (
            ("first admission", adm.groupby("subject_id")["admittime"].min()),
            ("first discharge", first["dischtime"]),
        ):
            anchor = anchor.reindex(cohort.subject_ids)

            merged = cohort.admissions.merge(
                anchor.rename("_a"), left_on="subject_id", right_index=True, how="left"
            )
            gap = (merged["admittime"] - merged["_a"]).dt.total_seconds() / 86_400
            readmit = merged.loc[(gap > 0) & (gap <= 30), "subject_id"].nunique()

            days = (dod - anchor).dt.total_seconds() / 86_400
            mort365 = int(((days >= 0) & (days <= 365)).sum())
            mort366 = int(((days >= 0) & (days <= 366)).sum())

            rows.append(
                {
                    "admission_set": name,
                    "anchor": anchor_name,
                    "n_admissions": len(adm),
                    "in_hospital_mortality_n": in_hosp,
                    "readmission_30d_n": readmit,
                    "readmission_30d_pct": round(readmit / n * 100, 1),
                    "mortality_365d_n": mort365,
                    "mortality_366d_n": mort366,
                    "mortality_366d_pct": round(mort366 / n * 100, 1),
                }
            )

    table = pd.DataFrame(rows)
    if paths is not None:
        table.to_csv(
            paths.ensure_results() / "sensitivity_outcome_definitions.csv", index=False
        )
    return table
