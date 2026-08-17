"""
Step 6: Symptom Co-occurrence Network Analysis
- Phi correlation matrix (phi >= 0.10 threshold)
- Louvain community detection (3 clusters)
- Louvain stability: 100 random seeds (ARI vs reference partition)
- Bootstrap stability: 200 patient resamples (ARI vs reference partition)
- Sensitivity analysis: phi >= 0.10, 0.15, 0.20

Requirements:
    pip install networkx python-louvain scikit-learn scipy matplotlib seaborn pandas numpy

Usage:
    python 06_network_analysis.py --method gemini
    python 06_network_analysis.py --method claude
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

# ── Symptom clusters (Louvain, seed=42, phi>=0.10) ────────────────────────
CLUSTER_MAP = {
    # Cluster 1: Constitutional / Systemic
    "lack_of_energy": 1, "pain": 1, "shortness_of_breath": 1,
    "problems_with_urination": 1, "feeling_drowsy": 1,
    "difficulty_sleeping": 1, "feeling_sad": 1,
    "worrying": 1, "feeling_nervous": 1,
    # Cluster 2: CRC Disease-Specific
    "diarrhoea": 2, "constipation": 2, "blood_in_stool": 2,
    "abdominal_pain": 2, "nausea": 2, "weight_loss": 2, "lack_of_appetite": 2,
    # Cluster 3: Gastrointestinal
    "flatulence_gas": 3, "frequent_bowel_movements": 3,
    "feeling_bloated": 3, "vomiting": 3, "dizziness": 3,
}

PRED_FILES = {
    "gemini": "files/Final/predictions_gemini_syn_full .csv",
    "claude": "files/Final/predictions_claude_syn_full .csv",
}


def load_patient_symptoms(pred_file: str, prevalence_threshold: float = 0.05):
    """Aggregate note-level predictions to patient level (any mention = present)."""
    df = pd.read_csv(pred_file)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    pt = df.groupby("subject_id")[pred_cols].max().reset_index()
    prev = pt[pred_cols].mean()
    included = prev[prev >= prevalence_threshold].index.tolist()
    print(f"  Patients: {len(pt)}, Symptoms >= {prevalence_threshold*100:.0f}%: {len(included)}")
    return pt, included


def build_graph(X: np.ndarray, threshold: float = 0.10) -> nx.Graph:
    """Build phi correlation network with given threshold."""
    n = X.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            if r >= threshold:
                G.add_edge(i, j, weight=float(r))
    return G


def louvain_stability(G: nx.Graph, n_sym: int, n_seeds: int = 100):
    """Run Louvain with 100 random seeds; compute ARI vs seed=42 reference."""
    ref = community_louvain.best_partition(G, random_state=42, weight="weight")
    ref_labels = [ref[i] for i in range(n_sym)]
    n_ref_clusters = len(set(ref_labels))

    aris, n_clusters = [], []
    for seed in range(n_seeds):
        part = community_louvain.best_partition(G, random_state=seed, weight="weight")
        labs = [part[i] for i in range(n_sym)]
        aris.append(adjusted_rand_score(ref_labels, labs))
        n_clusters.append(len(set(labs)))

    aris = np.array(aris)
    print(f"\n  === Louvain Stability ({n_seeds} seeds) ===")
    print(f"  Reference clusters (seed=42): {n_ref_clusters}")
    print(f"  Always same n_clusters: {sum(1 for n in n_clusters if n == n_ref_clusters)}/{n_seeds}")
    print(f"  Mean ARI: {aris.mean():.3f}  SD: {aris.std():.3f}")
    print(f"  Range:    {aris.min():.3f}–{aris.max():.3f}")
    return ref_labels, aris


def bootstrap_stability(X: np.ndarray, ref_labels: list, n_sym: int,
                        threshold: float = 0.10, n_boot: int = 200):
    """Bootstrap patient resampling stability analysis."""
    np.random.seed(0)
    n_pt = X.shape[0]
    aris, n_clusters = [], []

    for _ in range(n_boot):
        idx = np.random.choice(n_pt, n_pt, replace=True)
        G_b = build_graph(X[idx], threshold)
        part = community_louvain.best_partition(G_b, weight="weight")
        labs = [part[i] for i in range(n_sym)]
        aris.append(adjusted_rand_score(ref_labels, labs))
        n_clusters.append(len(set(labs)))

    aris = np.array(aris)
    n_ref = len(set(ref_labels))
    recovered = sum(1 for n in n_clusters if n == n_ref)
    print(f"\n  === Bootstrap Stability ({n_boot} resamples) ===")
    print(f"  {n_ref}-cluster recovered: {recovered}/{n_boot} ({recovered/n_boot*100:.1f}%)")
    print(f"  Mean ARI: {aris.mean():.3f}  SD: {aris.std():.3f}")
    print(f"  Range:    {aris.min():.3f}–{aris.max():.3f}  Median: {np.median(aris):.3f}")
    return aris


def sensitivity_analysis(X: np.ndarray, n_sym: int, thresholds=(0.10, 0.15, 0.20)):
    """Network sensitivity to phi threshold."""
    print(f"\n  === Sensitivity Analysis ===")
    print(f"  {'Threshold':>10} {'Edges':>8} {'Clusters':>10}")
    for thr in thresholds:
        G = build_graph(X, thr)
        part = community_louvain.best_partition(G, random_state=42, weight="weight")
        n_c = len(set(part.values()))
        print(f"  phi >= {thr:.2f}:   {G.number_of_edges():>6}   {n_c:>8}")


def plot_network(G: nx.Graph, labels: list, symptom_names: list,
                 X: np.ndarray, out_path: str):
    """Visualize symptom network with cluster colors."""
    cluster_colors = {1: "#4472C4", 2: "#ED7D31", 3: "#70AD47"}
    color_map = [cluster_colors.get(labels[i], "#999") for i in range(len(labels))]
    node_size = [X[:, i].mean() * 3000 + 200 for i in range(len(labels))]
    pos = nx.spring_layout(G, seed=42, k=2)
    weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_size, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.4, edge_color="#888")
    nx.draw_networkx_labels(G, pos, labels={i: symptom_names[i] for i in range(len(symptom_names))},
                             font_size=7)
    from matplotlib.patches import Patch
    legend = [Patch(color="#4472C4", label="C1: Systemic"),
              Patch(color="#ED7D31", label="C2: CRC Disease-Specific"),
              Patch(color="#70AD47", label="C3: Gastrointestinal")]
    plt.legend(handles=legend, loc="upper left", fontsize=9)
    plt.title("CRC Symptom Co-occurrence Network (phi >= 0.10)", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Network plot saved: {out_path}")


def main(method: str = "gemini", base_dir: str = "."):
    pred_file = str(Path(base_dir) / PRED_FILES[method])
    print(f"\n{'='*55}")
    print(f"Network Analysis — {method.upper()}")
    print(f"{'='*55}")

    pt, included = load_patient_symptoms(pred_file)
    X = pt[included].values
    n_sym = len(included)
    symptom_names = [s.replace("pred_", "").replace("_", " ").title() for s in included]

    # Build reference network
    G = build_graph(X, threshold=0.10)
    print(f"  Edges (phi>=0.10): {G.number_of_edges()}")

    # Louvain stability (100 seeds)
    ref_labels, louvain_aris = louvain_stability(G, n_sym, n_seeds=100)

    # Bootstrap stability (200 resamples)
    boot_aris = bootstrap_stability(X, ref_labels, n_sym, threshold=0.10, n_boot=200)

    # Sensitivity analysis
    sensitivity_analysis(X, n_sym, thresholds=(0.10, 0.15, 0.20))

    # Network plot
    out_png = str(Path(base_dir) / f"symptom_network_{method}.png")
    plot_network(G, ref_labels, symptom_names, X, out_png)

    print(f"\nDone. Run 07_predictive_validity.py for OR analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gemini", "claude"], default="gemini")
    parser.add_argument("--base_dir", default=".")
    args = parser.parse_args()
    main(method=args.method, base_dir=args.base_dir)
