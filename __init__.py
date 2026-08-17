"""Reproducible analysis pipeline for the CRC symptom-cluster study.

Public entry points
-------------------
>>> from analysis import Paths, load_cohort, load_predictions, patient_level_matrix
>>> paths = Paths()
>>> cohort = load_cohort(paths)
>>> preds = load_predictions(paths.preds_gemini, cohort.notes)
>>> matrix = patient_level_matrix(preds, cohort.subject_ids)

See ``run_analysis.py`` for the full pipeline and ``validate.py`` for the
regression test that pins every number reported in the manuscript.
"""

from .config import (
    OUTCOMES,
    PHI_THRESHOLD,
    PREVALENCE_THRESHOLD,
    PUBLISHED_CLUSTERS,
    RANDOM_STATE,
    OutcomeSpec,
    Paths,
    pretty_symptom,
)
from .data import Cohort, load_cohort, load_predictions, patient_level_matrix
from .figures import plot_phi_matrix
from .irr import build_table_s2, load_kappa_results, summarise_irr
from .network import SymptomNetwork, build_network, build_table4
from .predictive import (
    burden_vs_metastatic_prevalence,
    cluster_burden,
    fit_outcome_models,
    metastatic_sensitivity,
    stratified_by_metastatic,
)
from .sensitivity import (
    bootstrap_stability,
    first_note_vs_any_note,
    louvain_seed_stability,
    outcome_definition_grid,
    threshold_sensitivity,
)
from .supplementary import build_table_s1
from .table1 import build_table1

__version__ = "1.0.0"

__all__ = [
    "Cohort", "OutcomeSpec", "Paths", "SymptomNetwork",
    "OUTCOMES", "PHI_THRESHOLD", "PREVALENCE_THRESHOLD", "PUBLISHED_CLUSTERS",
    "RANDOM_STATE",
    "bootstrap_stability", "build_network", "build_table1", "build_table4",
    "build_table_s1", "build_table_s2", "burden_vs_metastatic_prevalence",
    "cluster_burden", "first_note_vs_any_note", "fit_outcome_models",
    "load_cohort", "load_kappa_results", "load_predictions",
    "louvain_seed_stability", "metastatic_sensitivity", "outcome_definition_grid",
    "patient_level_matrix", "plot_phi_matrix", "pretty_symptom",
    "stratified_by_metastatic", "summarise_irr", "threshold_sensitivity",
]
