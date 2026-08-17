"""Predictive validity of the symptom clusters (Section 3.7).

Primary models adjust for age and sex only. This is deliberate: the question is
whether cluster burden carries prognostic information that is available from the
discharge note itself, so covariates that sit upstream of the symptoms (disease
stage) are *not* in the primary model. Stage adjustment is reported separately as
a post hoc sensitivity analysis, and it is reported for all three outcomes so
that the effect of adding it is visible everywhere, not only where it helps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import CLUSTER_LABELS, PUBLISHED_CLUSTERS, Paths
from .data import Cohort

__all__ = ["cluster_burden", "fit_outcome_models", "metastatic_sensitivity",
           "stratified_by_metastatic", "burden_vs_metastatic_prevalence"]

CLUSTER_ORDER = ["Systemic", "CRC", "GI"]


def cluster_burden(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster burden = count of present symptoms within that cluster."""
    missing = [
        s for members in PUBLISHED_CLUSTERS.values() for s in members
        if s not in matrix.columns
    ]
    if missing:
        raise KeyError(f"prediction matrix is missing cluster members: {missing}")
    return pd.DataFrame(
        {name: matrix[members].sum(axis=1) for name, members in PUBLISHED_CLUSTERS.items()},
        index=matrix.index,
    )[CLUSTER_ORDER]


def _design(cohort: Cohort, matrix: pd.DataFrame, *, adjust_metastatic: bool) -> pd.DataFrame:
    X = cluster_burden(matrix).copy()
    X["age"] = cohort.age
    X["male"] = cohort.is_male
    if adjust_metastatic:
        X["metastatic"] = cohort.metastatic
    return X.astype(float)


def _fit(y: pd.Series, X: pd.DataFrame, outcome: str, model: str) -> pd.DataFrame:
    result = sm.Logit(y.astype(float), sm.add_constant(X)).fit(disp=0)
    odds = np.exp(result.params)
    ci = np.exp(result.conf_int())
    rows = []
    for term in X.columns:
        rows.append(
            {
                "outcome": outcome,
                "model": model,
                "term": CLUSTER_LABELS.get(term, term),
                "OR": round(float(odds[term]), 2),
                "CI_low": round(float(ci.loc[term, 0]), 2),
                "CI_high": round(float(ci.loc[term, 1]), 2),
                "p": float(result.pvalues[term]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["pseudo_r2"] = float(result.prsquared)
    frame.attrs["n_obs"] = int(result.nobs)
    return frame


def _outcome_map(cohort: Cohort) -> dict[str, pd.Series]:
    return {
        "In-hospital mortality": cohort.in_hospital_mortality,
        "30-day readmission": cohort.readmission_30d,
        "1-year mortality": cohort.mortality_1y,
    }


def fit_outcome_models(
    cohort: Cohort, matrix: pd.DataFrame, paths: Paths | None = None
) -> pd.DataFrame:
    """Primary models: cluster burdens + age + sex."""
    X = _design(cohort, matrix, adjust_metastatic=False)
    frames = [
        _fit(y, X, outcome, "age + sex") for outcome, y in _outcome_map(cohort).items()
    ]
    table = pd.concat(frames, ignore_index=True)
    if paths is not None:
        table.to_csv(paths.ensure_results() / "predictive_primary_models.csv", index=False)
    return table


def metastatic_sensitivity(
    cohort: Cohort, matrix: pd.DataFrame, paths: Paths | None = None
) -> pd.DataFrame:
    """Post hoc sensitivity: primary models + metastatic-disease adjustment."""
    X = _design(cohort, matrix, adjust_metastatic=True)
    frames = [
        _fit(y, X, outcome, "age + sex + metastatic")
        for outcome, y in _outcome_map(cohort).items()
    ]
    table = pd.concat(frames, ignore_index=True)
    if paths is not None:
        table.to_csv(
            paths.ensure_results() / "predictive_metastatic_sensitivity.csv", index=False
        )
    return table


def stratified_by_metastatic(cohort: Cohort, matrix: pd.DataFrame) -> pd.DataFrame:
    """1-year mortality models fitted separately within metastatic strata."""
    X_all = _design(cohort, matrix, adjust_metastatic=False)
    y_all = cohort.mortality_1y
    met = cohort.metastatic

    rows = []
    for value, label in ((1, "metastatic"), (0, "non-metastatic")):
        idx = met.index[met == value]
        y, X = y_all.reindex(idx), X_all.reindex(idx)
        result = sm.Logit(y.astype(float), sm.add_constant(X)).fit(disp=0)
        odds, ci = np.exp(result.params), np.exp(result.conf_int())
        for term in CLUSTER_ORDER:
            rows.append(
                {
                    "stratum": label,
                    "n": len(idx),
                    "n_events": int(y.sum()),
                    "term": CLUSTER_LABELS[term],
                    "OR": round(float(odds[term]), 2),
                    "CI_low": round(float(ci.loc[term, 0]), 2),
                    "CI_high": round(float(ci.loc[term, 1]), 2),
                    "p": float(result.pvalues[term]),
                }
            )
    return pd.DataFrame(rows)


def burden_vs_metastatic_prevalence(
    cohort: Cohort, matrix: pd.DataFrame, cluster: str = "CRC", max_burden: int = 4
) -> pd.DataFrame:
    """Metastatic prevalence and 1-year mortality by cluster-burden level.

    This is the table that rules out confounding-by-stage as the explanation for
    an inverse association: if higher burden went with *less* advanced disease,
    metastatic prevalence would fall across the rows. It rises.
    """
    burden = cluster_burden(matrix)[cluster]
    met, mort = cohort.metastatic, cohort.mortality_1y

    rows = []
    for level in range(max_burden + 1):
        mask = (burden >= max_burden) if level == max_burden else (burden == level)
        idx = burden.index[mask]
        if len(idx) == 0:
            continue
        rows.append(
            {
                "cluster": CLUSTER_LABELS[cluster],
                "burden": f"≥{max_burden}" if level == max_burden else str(level),
                "n": len(idx),
                "metastatic_pct": round(float(met.reindex(idx).mean() * 100), 1),
                "mortality_1y_pct": round(float(mort.reindex(idx).mean() * 100), 1),
            }
        )
    return pd.DataFrame(rows)
