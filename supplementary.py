"""Supplementary Table S1 — prevalence of all 46 symptoms by extraction method."""

from __future__ import annotations

import pandas as pd

from .config import PREVALENCE_THRESHOLD, Paths, pretty_symptom

__all__ = ["build_table_s1"]


def build_table_s1(
    gemini_matrix: pd.DataFrame,
    claude_matrix: pd.DataFrame,
    prevalence_threshold: float = PREVALENCE_THRESHOLD,
    paths: Paths | None = None,
) -> pd.DataFrame:
    """Patient-level prevalence and network-inclusion status for every symptom.

    Reporting all 46 rows (rather than only the excluded ones) makes the
    inclusion decision auditable in both directions and shows exactly which
    symptoms the two models disagree about.
    """
    if list(gemini_matrix.columns) != list(claude_matrix.columns):
        raise ValueError("the two prediction matrices have different symptom columns")

    n = len(gemini_matrix)
    rows = []
    for col in gemini_matrix.columns:
        g_n, c_n = int(gemini_matrix[col].sum()), int(claude_matrix[col].sum())
        g_pct, c_pct = g_n / n * 100, c_n / n * 100
        rows.append(
            {
                "symptom": pretty_symptom(col),
                "pred_column": col,
                "gemini_n": g_n,
                "gemini_pct": round(g_pct, 1),
                "gemini_retained": g_pct >= prevalence_threshold * 100,
                "claude_n": c_n,
                "claude_pct": round(c_pct, 1),
                "claude_retained": c_pct >= prevalence_threshold * 100,
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values("gemini_pct", ascending=False)
        .reset_index(drop=True)
    )
    table["discordant_inclusion"] = (
        table["gemini_retained"] != table["claude_retained"]
    )

    if paths is not None:
        table.to_csv(paths.ensure_results() / "table_s1_symptom_prevalence.csv", index=False)
    return table
