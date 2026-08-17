"""Central configuration for the CRC symptom-cluster analysis.

Every threshold, seed, and code list used anywhere in the manuscript lives here so
that a reader can audit the operational definitions in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Override with the CRC_DATA_DIR / CRC_RESULTS_DIR environment variables, or pass
# an explicit Paths() instance to the analysis functions.

DEFAULT_DATA_DIR = Path(os.environ.get("CRC_DATA_DIR", "data"))
DEFAULT_RESULTS_DIR = Path(os.environ.get("CRC_RESULTS_DIR", "results"))


@dataclass(frozen=True)
class Paths:
    """Filesystem layout.

    MIMIC-IV files are *not* redistributed with this repository. Download them
    from PhysioNet under a valid data use agreement and place them in ``data/``.
    """

    data_dir: Path = DEFAULT_DATA_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR

    # --- MIMIC-IV hosp module (from PhysioNet) ---
    @property
    def admissions(self) -> Path:
        return self.data_dir / "admissions.csv"

    @property
    def patients(self) -> Path:
        return self.data_dir / "patients.csv"

    @property
    def diagnoses_icd(self) -> Path:
        return self.data_dir / "diagnoses_icd.csv"

    # --- Derived / project files ---
    @property
    def cchpi(self) -> Path:
        """CRC discharge notes with the extracted CC/HPI section.

        Columns: note_id, subject_id, hadm_id, charttime, cc_hpi_text,
        icd_code, icd_version.
        """
        return self.data_dir / "crc_discharge_cchpi.csv"

    @property
    def preds_gemini(self) -> Path:
        """Note-level Gemini 3.5 Flash predictions (note_id + 46 ``pred_*`` cols)."""
        return self.data_dir / "preds_gemini_full.csv"

    @property
    def preds_claude(self) -> Path:
        """Note-level Claude Haiku predictions (note_id + 46 ``pred_*`` cols)."""
        return self.data_dir / "predictions_claude_haiku_full.csv"

    @property
    def kappa_workbook(self) -> Path:
        """200-note double-annotation workbook.

        Sheets: ``Summary``, ``Final_Consensus_200``, ``Kappa_Results``.
        """
        return self.data_dir / "200_Note_Adjudication_Kappa_Results.xlsx"

    def ensure_results(self) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir


# --------------------------------------------------------------------------- #
# Cohort definition
# --------------------------------------------------------------------------- #

#: ICD-9 prefixes for colorectal cancer (153.x colon, 154.x rectum/rectosigmoid).
ICD9_CRC_PREFIXES = ("153", "154")

#: ICD-10 prefixes for colorectal cancer (C18 colon, C19 rectosigmoid, C20 rectum).
ICD10_CRC_PREFIXES = ("C18", "C19", "C20")

#: Secondary/metastatic neoplasm codes used for the metastatic-disease flag.
#: Evaluated over **all** admissions of a cohort patient (not only CRC-coded ones),
#: which is what reproduces the n = 775 (51.4%) reported in Table 1.
ICD9_METASTATIC_PREFIXES = ("196", "197", "198", "199")
ICD10_METASTATIC_PREFIXES = ("C77", "C78", "C79", "C80")

#: Admission types counted as "emergency or urgent" in Table 1.
EMERGENT_ADMISSION_TYPES = ("EW EMER.", "DIRECT EMER.", "URGENT")

#: Admission types counted as "surgical admission type" in Table 1.
SURGICAL_ADMISSION_TYPES = ("SURGICAL SAME DAY ADMISSION", "ELECTIVE")


# --------------------------------------------------------------------------- #
# Outcome definitions (single source of truth — Table 1 and Section 3.7 agree)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OutcomeSpec:
    """Operational definitions for the three clinical outcomes.

    All three are computed over **every** admission of a cohort patient, anchored
    on that patient's first admission/discharge. Restricting the admission set to
    CRC-coded admissions, or re-anchoring on the first *analysed note*, changes the
    counts materially; see ``analysis/sensitivity.py::outcome_definition_grid``.
    """

    #: In-hospital mortality: ``hospital_expire_flag`` in any admission.
    #: 30-day readmission: any admission starting within this many days *after*
    #: the first discharge (strictly greater than 0 days).
    readmission_window_days: float = 30.0

    #: 1-year mortality: death within this many days of the first admission.
    #: 366 (rather than 365) is what reproduces the n = 233 in Table 1; the
    #: 365-day variant gives n = 231 and is reported as a sensitivity analysis.
    mortality_window_days: float = 366.0


OUTCOMES = OutcomeSpec()


# --------------------------------------------------------------------------- #
# Network analysis
# --------------------------------------------------------------------------- #

#: Minimum patient-level prevalence for a symptom to enter the network.
PREVALENCE_THRESHOLD = 0.05

#: Minimum phi correlation for an edge to be included.
PHI_THRESHOLD = 0.10

#: Reference seed for Louvain community detection.
RANDOM_STATE = 42

#: Thresholds swept in the phi-threshold sensitivity analysis.
PHI_THRESHOLD_GRID = (0.10, 0.15, 0.20)

#: Number of Louvain seeds in the reproducibility sensitivity analysis.
N_LOUVAIN_SEEDS = 100

#: Number of bootstrap resamples in the stability sensitivity analysis.
N_BOOTSTRAP = 200


# --------------------------------------------------------------------------- #
# Published cluster membership (Table 4, Gemini 3.5 Flash)
# --------------------------------------------------------------------------- #
# Ordered by strength centrality within each cluster, matching Table 4 exactly.
# ``network.detect_clusters`` re-derives these from the data; this mapping is the
# frozen published version used by the predictive-validity models so that those
# models cannot silently drift if the clustering is re-run.

PUBLISHED_CLUSTERS: dict[str, list[str]] = {
    "Systemic": [
        "pred_lack_of_appetite",
        "pred_lack_of_energy",
        "pred_weight_loss",
        "pred_shortness_of_breath",
        "pred_swelling_of_arms_or_legs",
        "pred_cough",
        "pred_changes_in_skin",
    ],
    "CRC": [
        "pred_diarrhoea",
        "pred_problems_with_urination",
        "pred_frequent_bowel_movements",
        "pred_dizziness",
        "pred_anal_rectal_pain",
        "pred_blood_in_stool",
    ],
    "GI": [
        "pred_pain",
        "pred_nausea",
        "pred_abdominal_pain",
        "pred_vomiting",
        "pred_constipation",
        "pred_feeling_bloated",
        "pred_flatulence_gas",
    ],
}

#: Human-readable cluster names as used in the manuscript.
CLUSTER_LABELS = {
    "Systemic": "Systemic Symptom Cluster",
    "CRC": "CRC Disease-Specific Symptom Cluster",
    "GI": "Gastrointestinal Symptom Cluster",
}

#: Table 4 node codes (A1-A7, B1-B6, C1-C7).
NODE_CODES: dict[str, str] = {
    key: f"{prefix}{i + 1}"
    for prefix, cluster in zip("ABC", ("Systemic", "CRC", "GI"))
    for i, key in enumerate(PUBLISHED_CLUSTERS[cluster])
}


def pretty_symptom(pred_column: str) -> str:
    """``pred_lack_of_appetite`` -> ``Lack of appetite``."""
    special = {
        "pred_i_dont_look_like_myself": "I don't look like myself",
        "pred_flatulence_gas": "Flatulence/gas",
        "pred_numbness_tingling_in_hands_feet": "Numbness/tingling in hands or feet",
        "pred_sore_skin_around_stoma_anal_area": "Sore skin around stoma/anal area",
        "pred_problems_with_sexual_interest_or_activity": (
            "Problems with sexual interest or activity"
        ),
    }
    if pred_column in special:
        return special[pred_column]
    text = pred_column.removeprefix("pred_").replace("_", " ")
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

#: Single-hue sequential blue ramp (light -> dark) for the phi heatmap. A
#: sequential ramp is correct here because no negative phi correlation survives
#: the prevalence filter; a diverging red/blue scale would imply polarity the
#: data do not contain.
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

INK = "#1a1a18"
MUTED = "#57564f"
SURFACE = "#ffffff"
DIAGONAL_FILL = "#e8e6e1"
