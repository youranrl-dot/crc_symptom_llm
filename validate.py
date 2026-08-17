#!/usr/bin/env python3
"""Regression test: pin every number reported in the manuscript.

If a refactor, a library upgrade, or a data refresh changes any published value,
this script fails loudly and names the value. Run it before every commit that
touches ``analysis/``.

    python validate.py                # full check
    python validate.py --fast         # skip the 200-iteration bootstrap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

from analysis import (
    Paths,
    bootstrap_stability,
    build_network,
    burden_vs_metastatic_prevalence,
    cluster_burden,
    first_note_vs_any_note,
    fit_outcome_models,
    load_cohort,
    load_kappa_results,
    load_predictions,
    louvain_seed_stability,
    metastatic_sensitivity,
    patient_level_matrix,
    summarise_irr,
    threshold_sensitivity,
)


@dataclass
class Check:
    name: str
    expected: object
    actual: object
    tol: float | None = None

    @property
    def passed(self) -> bool:
        if self.tol is None:
            return self.expected == self.actual
        try:
            return abs(float(self.actual) - float(self.expected)) <= self.tol
        except (TypeError, ValueError):
            return False


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, expected, actual, tol: float | None = None) -> None:
        self.checks.append(Check(name, expected, actual, tol))

    def render(self) -> int:
        section = None
        for check in self.checks:
            head = check.name.split(" | ")[0]
            if head != section:
                section = head
                print(f"\n{section}")
            label = check.name.split(" | ", 1)[-1]
            mark = "PASS" if check.passed else "FAIL"
            detail = f"expected {check.expected!r}, got {check.actual!r}"
            print(f"  [{mark}] {label:52s} {detail}")
        failed = [c for c in self.checks if not c.passed]
        print(f"\n{len(self.checks) - len(failed)}/{len(self.checks)} checks passed")
        if failed:
            print("FAILED: " + "; ".join(c.name for c in failed))
        return 1 if failed else 0


def _or_of(table, outcome: str, term_fragment: str) -> tuple[float, float, float]:
    row = table[
        (table["outcome"] == outcome) & table["term"].str.contains(term_fragment)
    ].iloc[0]
    return float(row["OR"]), float(row["CI_low"]), float(row["CI_high"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip the bootstrap check")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    paths = Paths(data_dir=args.data_dir) if args.data_dir else Paths()
    r = Report()

    # ---------------------------------------------------------------- cohort
    cohort = load_cohort(paths)
    r.add("Cohort | patients", 1507, cohort.n)
    r.add("Cohort | discharge notes", 2728, len(cohort.notes))
    r.add("Cohort | admissions (all)", 6677, len(cohort.admissions))
    r.add("Cohort | metastatic disease", 775, int(cohort.metastatic.sum()))

    # --------------------------------------------------------------- Table 1
    age = cohort.age
    r.add("Table 1 | mean age", 65.5, round(float(age.mean()), 1), 0.05)
    r.add("Table 1 | male", 774, int(cohort.is_male.sum()))
    r.add("Table 1 | admissions/patient (mean)", 4.4,
          round(float(cohort.n_admissions.mean()), 1), 0.05)
    r.add("Table 1 | admissions/patient (SD)", 4.2,
          round(float(cohort.n_admissions.std()), 1), 0.05)
    los = cohort.length_of_stay_days
    r.add("Table 1 | LOS mean", 5.4, round(float(los.mean()), 1), 0.05)
    r.add("Table 1 | emergency or urgent (any)", 1128, int(cohort.emergency_or_urgent.sum()))
    r.add("Table 1 | surgical admission type (any)", 940, int(cohort.surgical_admission.sum()))
    r.add("Table 1 | in-hospital mortality", 155, int(cohort.in_hospital_mortality.sum()))
    r.add("Table 1 | 30-day readmission", 409, int(cohort.readmission_30d.sum()))
    r.add("Table 1 | 1-year mortality", 233, int(cohort.mortality_1y.sum()))

    # ------------------------------------------------------------------- IRR
    irr = summarise_irr(load_kappa_results(paths))
    r.add("Section 3.2 | discordant pairs", 277, irr["n_discordant"])
    r.add("Section 3.2 | pooled agreement (%)", 97.0, irr["pooled_percent_agreement"], 0.05)
    r.add("Section 3.2 | macro kappa", 0.4854, irr["macro_kappa"], 0.0005)
    r.add("Section 3.2 | kappa >= 0.60", 20, irr["n_kappa_ge_060"])
    r.add("Section 3.2 | kappa < 0.20", 12, irr["n_kappa_lt_020"])
    r.add("Section 3.2 | not estimable", 5, irr["n_not_estimable"])

    # --------------------------------------------------------------- network
    gemini = load_predictions(paths.preds_gemini, cohort.notes)
    claude = load_predictions(paths.preds_claude, cohort.notes)
    gm = patient_level_matrix(gemini, cohort.subject_ids)
    cm = patient_level_matrix(claude, cohort.subject_ids)

    net = build_network(gm)
    offdiag = net.offdiagonal_phi()
    r.add("Section 3.5 | Gemini symptoms retained", 20, len(net.retained))
    r.add("Section 3.5 | Gemini edges", 136, net.n_edges)
    r.add("Section 3.5 | Gemini communities", 3, net.n_communities)
    r.add("Section 3.5 | Gemini cluster sizes", [7, 7, 6], net.community_sizes)
    r.add("Section 3.5 | phi min", 0.006, round(float(offdiag.min()), 3), 0.0005)
    r.add("Section 3.5 | phi max", 0.760, round(float(offdiag.max()), 3), 0.0005)

    cnet = build_network(cm)
    r.add("Section 3.5 | Claude symptoms retained", 21, len(cnet.retained))
    r.add("Section 3.5 | Claude cluster sizes", [10, 7, 4], cnet.community_sizes)

    # ----------------------------------------------------------- sensitivity
    thresh = threshold_sensitivity(gm, net).set_index("phi_threshold")
    r.add("Section 3.6 | edges @ phi>=0.15", 88, int(thresh.loc[0.15, "n_edges"]))
    r.add("Section 3.6 | edges @ phi>=0.20", 51, int(thresh.loc[0.20, "n_edges"]))
    r.add("Section 3.6 | clusters @ phi>=0.20", 5, int(thresh.loc[0.20, "n_clusters"]))
    r.add("Section 3.6 | ARI 0.10 vs 0.15", 0.438,
          float(thresh.loc[0.15, "ari_vs_reference"]), 0.002)

    seeds = louvain_seed_stability(net)
    r.add("Section 3.6 | 3 clusters across 100 seeds", 100, seeds["n_three_cluster"])
    r.add("Section 3.6 | seed ARI mean", 0.985, seeds["ari_mean"], 0.002)

    note_summary, _ = first_note_vs_any_note(gemini, cohort.subject_ids)
    r.add("Section 3.6 | mean symptoms (any note)", 4.15,
          note_summary["mean_symptoms_any_note"], 0.005)
    r.add("Section 3.6 | mean symptoms (first note)", 2.73,
          note_summary["mean_symptoms_first_note"], 0.005)

    if not args.fast:
        boot = bootstrap_stability(gm, net)
        r.add("Section 3.6 | bootstrap 3-cluster runs", 192, boot["n_three_cluster"])
        r.add("Section 3.6 | bootstrap ARI mean", 0.733, boot["ari_mean"], 0.002)

    # ---------------------------------------------------------- predictive
    primary = fit_outcome_models(cohort, gm)
    for outcome, term, expected in (
        ("In-hospital mortality", "Systemic", (1.33, 1.18, 1.49)),
        ("In-hospital mortality", "Gastrointestinal", (1.13, 1.02, 1.25)),
        ("In-hospital mortality", "CRC Disease", (1.01, 0.87, 1.17)),
        ("30-day readmission", "CRC Disease", (1.20, 1.08, 1.34)),
        ("30-day readmission", "Systemic", (1.09, 0.99, 1.19)),
        ("30-day readmission", "Gastrointestinal", (1.02, 0.96, 1.10)),
        ("1-year mortality", "Systemic", (1.41, 1.27, 1.56)),
        ("1-year mortality", "Gastrointestinal", (1.11, 1.02, 1.21)),
        ("1-year mortality", "CRC Disease", (0.85, 0.74, 0.98)),
    ):
        r.add(f"Section 3.7 | {outcome}: {term}", expected, _or_of(primary, outcome, term),
              None)

    met = metastatic_sensitivity(cohort, gm)
    r.add("Section 3.7 | metastatic-adj 1y: CRC", (0.83, 0.71, 0.95),
          _or_of(met, "1-year mortality", "CRC Disease"))
    r.add("Section 3.7 | metastatic-adj 1y: Systemic", (1.32, 1.19, 1.48),
          _or_of(met, "1-year mortality", "Systemic"))
    r.add("Section 3.7 | metastatic-adj 1y: GI", (1.01, 0.92, 1.11),
          _or_of(met, "1-year mortality", "Gastrointestinal"))

    # ------------------------------------------------------- benchmark (Tables 2-3)
    from analysis import Paths as _P  # noqa: F401
    import subprocess, tempfile, csv as _csv
    bench = Path(tempfile.mkdtemp())
    hyb = {}
    for name, src in (("claude", paths.preds_claude), ("gemini", paths.preds_gemini)):
        out = bench / f"{name}_hybrid.csv"
        subprocess.run([sys.executable, str(Path(__file__).parent / "run_hybrid.py"),
                        "--pred", str(src), "--notes", str(paths.cchpi),
                        "--out", str(out)], check=True, capture_output=True)
        hyb[name] = out
    cmd = [sys.executable, str(Path(__file__).parent / "compare_with_gold.py"),
           "--gold", str(paths.kappa_workbook), "--na-policy", "zero",
           "--out", str(bench),
           "--pred", f"claude={paths.preds_claude}",
           "--pred", f"gemini={paths.preds_gemini}",
           "--pred", f"claude_hybrid={hyb['claude']}",
           "--pred", f"gemini_hybrid={hyb['gemini']}"]
    subprocess.run(cmd, check=True, capture_output=True)
    bt = {row["method"]: row for row in
          _csv.DictReader(open(bench / "table2_table3_summary.csv"))}
    for meth, key, expected in (
        ("claude", "macro_f1", 0.6291), ("gemini", "macro_f1", 0.6533),
        ("claude_hybrid", "macro_f1", 0.5262), ("gemini_hybrid", "macro_f1", 0.5437),
    ):
        r.add(f"Table 2 | {meth} Macro F1", expected, round(float(bt[meth][key]), 4), 0.0002)
    for meth, expected in (
        ("claude", (495, 176, 53, 8476)), ("gemini", (489, 164, 59, 8488)),
        ("claude_hybrid", (358, 120, 190, 8532)), ("gemini_hybrid", (349, 103, 199, 8549)),
    ):
        got = tuple(int(bt[meth][c]) for c in ("TP", "FP", "FN", "TN"))
        r.add(f"Table 3 | {meth} TP/FP/FN/TN", expected, got)

    grad = burden_vs_metastatic_prevalence(cohort, gm).set_index("burden")
    r.add("Section 3.7 | metastatic %% at CRC burden 0", 43.5,
          float(grad.loc["0", "metastatic_pct"]), 0.05)
    r.add("Section 3.7 | metastatic %% at CRC burden 3", 78.8,
          float(grad.loc["3", "metastatic_pct"]), 0.05)

    return r.render()


if __name__ == "__main__":
    sys.exit(main())
