"""Loading, cohort construction, and outcome derivation.

The cohort is the set of patients who (a) carry a colorectal-cancer ICD code in
the MIMIC-IV hosp module and (b) have at least one discharge note from which a
CC/HPI section could be extracted. That set is 1,507 patients / 2,728 notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .config import (
    EMERGENT_ADMISSION_TYPES,
    ICD9_CRC_PREFIXES,
    ICD9_METASTATIC_PREFIXES,
    ICD10_CRC_PREFIXES,
    ICD10_METASTATIC_PREFIXES,
    OUTCOMES,
    SURGICAL_ADMISSION_TYPES,
    OutcomeSpec,
    Paths,
)

__all__ = ["Cohort", "load_cohort", "load_predictions", "patient_level_matrix"]


def _prefix_mask(codes: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    """Case-insensitive ``startswith`` over a tuple of prefixes."""
    normalised = codes.astype(str).str.strip().str.upper()
    return normalised.str.startswith(prefixes)


@dataclass
class Cohort:
    """The analysis cohort plus everything derived from MIMIC-IV structured data.

    Attributes
    ----------
    subject_ids
        Sorted list of the 1,507 cohort ``subject_id`` values. Every per-patient
        Series produced by this class is reindexed onto this order, so downstream
        code can concatenate freely without alignment bugs.
    notes
        The 2,728 analysed discharge notes.
    admissions
        *All* admissions belonging to cohort patients, sorted by
        ``(subject_id, admittime)``.
    """

    subject_ids: list[int]
    notes: pd.DataFrame
    admissions: pd.DataFrame
    patients: pd.DataFrame
    crc_hadm_ids: set[int]
    metastatic_subject_ids: set[int]
    outcome_spec: OutcomeSpec = OUTCOMES

    # ------------------------------------------------------------------ #
    # Basics
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.subject_ids)

    def _reindex(self, s: pd.Series, fill=0) -> pd.Series:
        return s.reindex(self.subject_ids).fillna(fill)

    @cached_property
    def first_admission(self) -> pd.DataFrame:
        """First admission row per patient (by ``admittime``)."""
        return self.admissions.groupby("subject_id").first()

    @cached_property
    def date_of_death(self) -> pd.Series:
        return (
            self.patients.set_index("subject_id")["dod"].reindex(self.subject_ids)
        )

    # ------------------------------------------------------------------ #
    # Demographics
    # ------------------------------------------------------------------ #
    @cached_property
    def age(self) -> pd.Series:
        return self.patients.set_index("subject_id")["anchor_age"].reindex(
            self.subject_ids
        )

    @cached_property
    def is_male(self) -> pd.Series:
        gender = self.patients.set_index("subject_id")["gender"].reindex(
            self.subject_ids
        )
        return (gender == "M").astype(int)

    def first_admission_attribute(self, column: str) -> pd.Series:
        """Value of ``column`` recorded on the patient's first admission.

        Race, insurance, and marital status in Table 1 all use this rule. Applying
        one consistent rule matters: attributing marital status from the first
        *note-bearing* admission instead shifts four of the five categories.
        """
        return self.first_admission[column].reindex(self.subject_ids)

    # ------------------------------------------------------------------ #
    # Comorbidity / disease characteristics
    # ------------------------------------------------------------------ #
    @cached_property
    def metastatic(self) -> pd.Series:
        """Binary metastatic-disease flag (any admission)."""
        return pd.Series(
            [int(s in self.metastatic_subject_ids) for s in self.subject_ids],
            index=self.subject_ids,
            name="metastatic",
        )

    # ------------------------------------------------------------------ #
    # Hospitalisation descriptors
    # ------------------------------------------------------------------ #
    @cached_property
    def n_admissions(self) -> pd.Series:
        return self._reindex(self.admissions.groupby("subject_id").size())

    @cached_property
    def length_of_stay_days(self) -> pd.Series:
        """Per-admission length of stay (one row per admission, not per patient)."""
        delta = self.admissions["dischtime"] - self.admissions["admittime"]
        return delta.dt.total_seconds() / 86_400

    def any_admission_type(self, types: tuple[str, ...]) -> pd.Series:
        """1 if any of the patient's admissions has one of ``types``."""
        flag = self.admissions["admission_type"].astype(str).str.upper().isin(
            [t.upper() for t in types]
        )
        per_patient = (
            self.admissions.assign(_flag=flag).groupby("subject_id")["_flag"].max()
        )
        return self._reindex(per_patient, fill=False).astype(int)

    @cached_property
    def emergency_or_urgent(self) -> pd.Series:
        return self.any_admission_type(EMERGENT_ADMISSION_TYPES)

    @cached_property
    def surgical_admission(self) -> pd.Series:
        return self.any_admission_type(SURGICAL_ADMISSION_TYPES)

    # ------------------------------------------------------------------ #
    # Outcomes — the single source of truth for Table 1 *and* Section 3.7
    # ------------------------------------------------------------------ #
    @cached_property
    def in_hospital_mortality(self) -> pd.Series:
        """Death during any admission."""
        per_patient = self.admissions.groupby("subject_id")["hospital_expire_flag"].max()
        return self._reindex(per_patient).astype(int).rename("in_hospital_mortality")

    @cached_property
    def readmission_30d(self) -> pd.Series:
        """Any admission beginning within the window after the *first* discharge."""
        anchor = self.first_admission["dischtime"].rename("_anchor")
        merged = self.admissions.merge(
            anchor, left_on="subject_id", right_index=True, how="left"
        )
        gap_days = (merged["admittime"] - merged["_anchor"]).dt.total_seconds() / 86_400
        window = self.outcome_spec.readmission_window_days
        hit = merged.loc[(gap_days > 0) & (gap_days <= window), "subject_id"].unique()
        hit_set = set(hit)
        return pd.Series(
            [int(s in hit_set) for s in self.subject_ids],
            index=self.subject_ids,
            name="readmission_30d",
        )

    @cached_property
    def mortality_1y(self) -> pd.Series:
        """Death within the window of the *first* admission."""
        first_admit = (
            self.admissions.groupby("subject_id")["admittime"].min().reindex(
                self.subject_ids
            )
        )
        days = (self.date_of_death - first_admit).dt.total_seconds() / 86_400
        window = self.outcome_spec.mortality_window_days
        return ((days >= 0) & (days <= window)).astype(int).rename("mortality_1y")

    def outcomes(self) -> pd.DataFrame:
        return pd.concat(
            [self.in_hospital_mortality, self.readmission_30d, self.mortality_1y],
            axis=1,
        )


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_cohort(paths: Paths | None = None, outcome_spec: OutcomeSpec | None = None) -> Cohort:
    """Build the analysis cohort from the MIMIC-IV files listed in ``paths``."""
    paths = paths or Paths()
    spec = outcome_spec or OUTCOMES

    notes = pd.read_csv(paths.cchpi)
    subject_ids = sorted(notes["subject_id"].unique().tolist())

    patients = pd.read_csv(paths.patients)
    patients["dod"] = pd.to_datetime(patients["dod"])

    admissions = pd.read_csv(
        paths.admissions, parse_dates=["admittime", "dischtime", "deathtime"]
    )
    admissions = (
        admissions[admissions["subject_id"].isin(subject_ids)]
        .sort_values(["subject_id", "admittime"])
        .reset_index(drop=True)
    )

    diagnoses = pd.read_csv(paths.diagnoses_icd, dtype={"icd_code": str})
    cohort_dx = diagnoses[diagnoses["subject_id"].isin(subject_ids)]

    is_crc = (
        (cohort_dx["icd_version"] == 9)
        & _prefix_mask(cohort_dx["icd_code"], ICD9_CRC_PREFIXES)
    ) | (
        (cohort_dx["icd_version"] == 10)
        & _prefix_mask(cohort_dx["icd_code"], ICD10_CRC_PREFIXES)
    )
    crc_hadm_ids = set(cohort_dx.loc[is_crc, "hadm_id"].unique().tolist())

    is_met = (
        (cohort_dx["icd_version"] == 9)
        & _prefix_mask(cohort_dx["icd_code"], ICD9_METASTATIC_PREFIXES)
    ) | (
        (cohort_dx["icd_version"] == 10)
        & _prefix_mask(cohort_dx["icd_code"], ICD10_METASTATIC_PREFIXES)
    )
    metastatic_ids = set(cohort_dx.loc[is_met, "subject_id"].unique().tolist())

    return Cohort(
        subject_ids=subject_ids,
        notes=notes,
        admissions=admissions,
        patients=patients,
        crc_hadm_ids=crc_hadm_ids,
        metastatic_subject_ids=metastatic_ids,
        outcome_spec=spec,
    )


def load_predictions(path, notes: pd.DataFrame) -> pd.DataFrame:
    """Read a note-level prediction file and attach ``subject_id``/``charttime``."""
    preds = pd.read_csv(path)
    keys = ["note_id", "subject_id"]
    if "charttime" in notes.columns:
        keys.append("charttime")
    merged = preds.merge(notes[keys], on="note_id", how="left")
    if merged["subject_id"].isna().any():
        missing = int(merged["subject_id"].isna().sum())
        raise ValueError(f"{missing} prediction rows could not be matched to a note")
    return merged


def symptom_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("pred_")]


def patient_level_matrix(
    preds: pd.DataFrame, subject_ids: list[int], *, first_note_only: bool = False
) -> pd.DataFrame:
    """Aggregate note-level predictions to a patient x symptom binary matrix.

    Parameters
    ----------
    first_note_only
        If True, use only each patient's earliest note by ``charttime`` (the
        Section 3.6 sensitivity analysis) instead of the any-note maximum.
    """
    cols = symptom_columns(preds)
    if first_note_only:
        if "charttime" not in preds.columns:
            raise ValueError("first_note_only requires a charttime column")
        ordered = preds.sort_values(["subject_id", "charttime"])
        matrix = ordered.groupby("subject_id")[cols].first()
    else:
        matrix = preds.groupby("subject_id")[cols].max()
    return matrix.reindex(subject_ids).fillna(0).astype(int)
