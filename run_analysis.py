#!/usr/bin/env python3
"""Run the full CRC symptom-cluster analysis and write every table to ``results/``.

Usage
-----
    python run_analysis.py                     # everything
    python run_analysis.py --skip bootstrap    # skip the slow bootstrap
    python run_analysis.py --data-dir /path/to/mimic --results-dir out/

MIMIC-IV files are not redistributed here; see README for the expected layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from analysis import (
    Paths,
    bootstrap_stability,
    build_network,
    build_table1,
    build_table4,
    build_table_s1,
    build_table_s2,
    burden_vs_metastatic_prevalence,
    first_note_vs_any_note,
    fit_outcome_models,
    load_cohort,
    load_kappa_results,
    load_predictions,
    louvain_seed_stability,
    metastatic_sensitivity,
    outcome_definition_grid,
    patient_level_matrix,
    plot_phi_matrix,
    stratified_by_metastatic,
    summarise_irr,
    threshold_sensitivity,
)

STEPS = (
    "cohort", "table1", "irr", "network", "sensitivity", "bootstrap",
    "predictive", "figures", "supplementary",
)


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--skip", nargs="*", default=[], choices=STEPS,
                        help="steps to skip (bootstrap is the slow one)")
    args = parser.parse_args(argv)

    kwargs = {}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    if args.results_dir:
        kwargs["results_dir"] = args.results_dir
    paths = Paths(**kwargs)
    results = paths.ensure_results()
    skip = set(args.skip)
    summary: dict[str, object] = {}

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 40)

    # ------------------------------------------------------------------ cohort
    _rule("1. Cohort")
    cohort = load_cohort(paths)
    print(f"patients: {cohort.n:,}   notes: {len(cohort.notes):,}   "
          f"admissions: {len(cohort.admissions):,}")
    summary["cohort"] = {
        "n_patients": cohort.n,
        "n_notes": int(len(cohort.notes)),
        "n_admissions": int(len(cohort.admissions)),
        "n_metastatic": int(cohort.metastatic.sum()),
    }

    gemini = load_predictions(paths.preds_gemini, cohort.notes)
    claude = load_predictions(paths.preds_claude, cohort.notes)
    gemini_matrix = patient_level_matrix(gemini, cohort.subject_ids)
    claude_matrix = patient_level_matrix(claude, cohort.subject_ids)

    # ------------------------------------------------------------------ table 1
    if "table1" not in skip:
        _rule("2. Table 1 — patient characteristics")
        print(build_table1(cohort, paths).to_string(index=False))

    # ---------------------------------------------------------------------- IRR
    if "irr" not in skip:
        _rule("3. Inter-rater reliability")
        irr = summarise_irr(load_kappa_results(paths))
        for key, value in irr.items():
            print(f"  {key:28s} {value}")
        build_table_s2(paths)
        summary["inter_rater"] = irr

    # ------------------------------------------------------------------ network
    _rule("4. Symptom network (Gemini 3.5 Flash)")
    network = build_network(gemini_matrix)
    offdiag = network.offdiagonal_phi()
    print(f"retained symptoms : {len(network.retained)} of {gemini_matrix.shape[1]}")
    print(f"edges (phi >= {network.phi_threshold:.2f}) : {network.n_edges}")
    print(f"communities       : {network.n_communities}  sizes {network.community_sizes}")
    print(f"phi range         : {offdiag.min():.3f} to {offdiag.max():.3f}")
    print(build_table4(network, paths).to_string(index=False))
    summary["network"] = {
        "n_retained": len(network.retained),
        "n_edges": network.n_edges,
        "n_communities": network.n_communities,
        "community_sizes": network.community_sizes,
        "phi_min": round(float(offdiag.min()), 3),
        "phi_max": round(float(offdiag.max()), 3),
    }

    claude_network = build_network(claude_matrix)
    print(f"\nClaude Haiku network: {len(claude_network.retained)} symptoms, "
          f"{claude_network.n_edges} edges, {claude_network.n_communities} communities "
          f"{claude_network.community_sizes}")
    summary["network_claude"] = {
        "n_retained": len(claude_network.retained),
        "n_edges": claude_network.n_edges,
        "community_sizes": claude_network.community_sizes,
    }

    # -------------------------------------------------------------- sensitivity
    if "sensitivity" not in skip:
        _rule("5. Sensitivity analyses")
        print("-- phi threshold --")
        print(threshold_sensitivity(gemini_matrix, network, paths).to_string(index=False))

        print("\n-- Louvain seeds --")
        seeds = louvain_seed_stability(network)
        print("  " + "  ".join(f"{k}={v}" for k, v in seeds.items()))
        summary["louvain_seeds"] = seeds

        print("\n-- first-note vs any-note --")
        note_summary, dropped = first_note_vs_any_note(gemini, cohort.subject_ids,
                                                       paths=paths)
        print("  " + "  ".join(f"{k}={v}" for k, v in note_summary.items()))
        print(dropped.to_string(index=False))
        summary["first_vs_any_note"] = note_summary

        print("\n-- outcome definition grid --")
        print(outcome_definition_grid(cohort, paths).to_string(index=False))

    if "bootstrap" not in skip:
        _rule("6. Bootstrap cluster stability (slow)")
        boot = bootstrap_stability(gemini_matrix, network)
        print("  " + "  ".join(f"{k}={v}" for k, v in boot.items()))
        summary["bootstrap"] = boot

    # --------------------------------------------------------------- predictive
    if "predictive" not in skip:
        _rule("7. Predictive validity")
        primary = fit_outcome_models(cohort, gemini_matrix, paths)
        print(primary.to_string(index=False))

        print("\n-- metastatic-adjusted sensitivity --")
        print(metastatic_sensitivity(cohort, gemini_matrix, paths).to_string(index=False))

        print("\n-- 1-year mortality stratified by metastatic status --")
        print(stratified_by_metastatic(cohort, gemini_matrix).to_string(index=False))

        print("\n-- CRC cluster burden vs metastatic prevalence --")
        print(burden_vs_metastatic_prevalence(cohort, gemini_matrix).to_string(index=False))

        summary["outcomes"] = {
            "in_hospital_mortality_n": int(cohort.in_hospital_mortality.sum()),
            "readmission_30d_n": int(cohort.readmission_30d.sum()),
            "mortality_1y_n": int(cohort.mortality_1y.sum()),
        }

    # ------------------------------------------------------------- supplementary
    if "supplementary" not in skip:
        _rule("8. Supplementary Table S1")
        s1 = build_table_s1(gemini_matrix, claude_matrix, paths=paths)
        print(s1[s1["discordant_inclusion"]].to_string(index=False))

    if "figures" not in skip:
        _rule("9. Figures")
        print("wrote", plot_phi_matrix(network, paths=paths))

    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll outputs written to {results.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
