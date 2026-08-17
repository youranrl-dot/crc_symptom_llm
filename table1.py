"""Table 1 — demographic and clinical characteristics of the study cohort."""

from __future__ import annotations

import pandas as pd

from .config import Paths
from .data import Cohort

__all__ = ["build_table1"]


def _race_group(value: str) -> str:
    text = str(value).upper()
    if "WHITE" in text:
        return "White"
    if "BLACK" in text or "AFRICAN" in text:
        return "Black/African American"
    if "ASIAN" in text:
        return "Asian"
    if "HISPANIC" in text or "LATINO" in text:
        return "Hispanic/Latino"
    return "Other/Unknown"


def _insurance_group(value: str) -> str:
    text = str(value).strip()
    return text if text in {"Medicare", "Private", "Medicaid"} else "Other/No charge"


def _marital_group(value) -> str:
    text = str(value).strip().upper()
    mapping = {
        "MARRIED": "Married/Partnered",
        "SINGLE": "Single",
        "DIVORCED": "Divorced/Separated",
        "WIDOWED": "Widowed",
    }
    return mapping.get(text, "Unknown")


def _count_row(label: str, n: int, total: int) -> dict:
    return {"Characteristic": label, "Value": f"{n:,} ({n / total * 100:.1f})"}


def build_table1(cohort: Cohort, paths: Paths | None = None) -> pd.DataFrame:
    """Return Table 1 as a two-column DataFrame and optionally write it to disk."""
    n = cohort.n
    rows: list[dict] = []

    def add(label: str, value: str) -> None:
        rows.append({"Characteristic": label, "Value": value})

    def add_counts(series: pd.Series, order: list[str]) -> None:
        counts = series.value_counts()
        for level in order:
            rows.append(_count_row(f"  {level}", int(counts.get(level, 0)), n))

    # --- Demographics -----------------------------------------------------
    add("Demographics", "")
    age = cohort.age
    add(
        "  Age, mean ± SD (median [IQR]), years",
        f"{age.mean():.1f} ± {age.std():.1f} "
        f"({age.median():.0f} [{age.quantile(.25):.0f}–{age.quantile(.75):.0f}])",
    )
    add("Sex, n (%)", "")
    n_male = int(cohort.is_male.sum())
    rows.append(_count_row("  Male", n_male, n))
    rows.append(_count_row("  Female", n - n_male, n))

    add("Race/ethnicity, n (%)", "")
    add_counts(
        cohort.first_admission_attribute("race").map(_race_group),
        ["White", "Black/African American", "Asian", "Hispanic/Latino", "Other/Unknown"],
    )

    add("Insurance, n (%)", "")
    add_counts(
        cohort.first_admission_attribute("insurance").map(_insurance_group),
        ["Medicare", "Private", "Medicaid", "Other/No charge"],
    )

    add("Marital status, n (%)", "")
    add_counts(
        cohort.first_admission_attribute("marital_status").map(_marital_group),
        ["Married/Partnered", "Single", "Divorced/Separated", "Widowed", "Unknown"],
    )

    # --- Disease ----------------------------------------------------------
    add("CRC characteristics", "")
    rows.append(
        _count_row("  Metastatic disease (any admission)", int(cohort.metastatic.sum()), n)
    )

    # --- Hospitalisation --------------------------------------------------
    add("Hospitalisation", "")
    n_adm = cohort.n_admissions
    add(
        "  No. of hospital admissions per patient, mean ± SD (median)",
        f"{n_adm.mean():.1f} ± {n_adm.std():.1f} ({n_adm.median():.0f})",
    )
    los = cohort.length_of_stay_days
    add(
        "  Length of stay per admission, mean ± SD (median), days",
        f"{los.mean():.1f} ± {los.std():.1f} ({los.median():.1f})",
    )
    rows.append(
        _count_row("  Surgical admission type (any)", int(cohort.surgical_admission.sum()), n)
    )
    rows.append(
        _count_row(
            "  Emergency or urgent admission (any)",
            int(cohort.emergency_or_urgent.sum()),
            n,
        )
    )

    # --- Outcomes ---------------------------------------------------------
    add("Clinical outcomes", "")
    rows.append(
        _count_row(
            "  In-hospital mortality (any admission)",
            int(cohort.in_hospital_mortality.sum()),
            n,
        )
    )
    rows.append(
        _count_row(
            "  30-day readmission (from first discharge)",
            int(cohort.readmission_30d.sum()),
            n,
        )
    )
    rows.append(
        _count_row(
            "  1-year mortality (from first admission)", int(cohort.mortality_1y.sum()), n
        )
    )

    table = pd.DataFrame(rows)
    table.columns = ["Characteristic", f"N = {n:,}"]

    if paths is not None:
        out = paths.ensure_results() / "table1_patient_characteristics.csv"
        table.to_csv(out, index=False)
    return table
