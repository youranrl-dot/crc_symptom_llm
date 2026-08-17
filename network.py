"""Symptom co-occurrence network: phi correlations, Louvain clusters, Table 4."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

try:  # python-louvain, the library named in the manuscript
    import community as community_louvain
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "python-louvain is required (pip install python-louvain). "
        "networkx's built-in louvain_communities gives slightly different ARI values."
    ) from exc

from .config import (
    NODE_CODES,
    PHI_THRESHOLD,
    PREVALENCE_THRESHOLD,
    RANDOM_STATE,
    Paths,
    pretty_symptom,
)

__all__ = ["SymptomNetwork", "build_network", "build_table4"]


@dataclass
class SymptomNetwork:
    """A thresholded symptom co-occurrence network and its Louvain partition."""

    matrix: pd.DataFrame          # patients x symptoms, binary
    retained: list[str]           # symptoms passing the prevalence filter
    excluded: list[str]
    phi: pd.DataFrame             # full phi correlation matrix over ``retained``
    graph: nx.Graph
    partition: dict[str, int]     # symptom -> community id
    phi_threshold: float
    prevalence_threshold: float
    random_state: int

    @property
    def n_patients(self) -> int:
        return len(self.matrix)

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def n_communities(self) -> int:
        return len(set(self.partition.values()))

    @property
    def community_sizes(self) -> list[int]:
        return sorted(pd.Series(self.partition).value_counts().tolist(), reverse=True)

    @property
    def prevalence(self) -> pd.Series:
        return self.matrix.mean()

    def strength_centrality(self) -> pd.Series:
        """Sum of absolute edge weights incident on each node."""
        values = {
            node: sum(abs(d["weight"]) for _, _, d in self.graph.edges(node, data=True))
            for node in self.graph.nodes
        }
        return pd.Series(values).reindex(self.retained)

    def offdiagonal_phi(self) -> pd.Series:
        mask = ~np.eye(len(self.retained), dtype=bool)
        return self.phi.where(mask).stack()

    def singletons(self) -> list[str]:
        counts = pd.Series(self.partition).value_counts()
        return [n for n, c in self.partition.items() if counts[c] == 1]

    def labels_for(self, symptoms: list[str]) -> list[int]:
        return [self.partition[s] for s in symptoms]


def build_network(
    matrix: pd.DataFrame,
    *,
    phi_threshold: float = PHI_THRESHOLD,
    prevalence_threshold: float = PREVALENCE_THRESHOLD,
    random_state: int = RANDOM_STATE,
) -> SymptomNetwork:
    """Build the phi network and run Louvain community detection.

    Edges are included when ``phi >= phi_threshold``. Because no negative phi
    correlation survives the prevalence filter in this cohort, the absolute-value
    convention used for strength centrality is a no-op here; it is retained so the
    code stays correct on data where negative pairs do occur.
    """
    prevalence = matrix.mean()
    retained = prevalence[prevalence >= prevalence_threshold].index.tolist()
    excluded = [c for c in matrix.columns if c not in retained]

    phi = matrix[retained].corr()

    graph = nx.Graph()
    graph.add_nodes_from(retained)
    for i, a in enumerate(retained):
        for b in retained[i + 1:]:
            weight = float(phi.loc[a, b])
            if weight >= phi_threshold:
                graph.add_edge(a, b, weight=weight)

    partition = community_louvain.best_partition(
        graph, weight="weight", random_state=random_state
    )

    return SymptomNetwork(
        matrix=matrix,
        retained=retained,
        excluded=excluded,
        phi=phi,
        graph=graph,
        partition=partition,
        phi_threshold=phi_threshold,
        prevalence_threshold=prevalence_threshold,
        random_state=random_state,
    )


def build_table4(network: SymptomNetwork, paths: Paths | None = None) -> pd.DataFrame:
    """Table 4 — cluster membership, prevalence and strength centrality.

    Rows are ordered by cluster, then by descending strength centrality, and are
    tagged with the A1–A7 / B1–B6 / C1–C7 codes used in the figures.
    """
    centrality = network.strength_centrality()
    prevalence = network.prevalence

    rows = []
    for symptom in network.retained:
        rows.append(
            {
                "code": NODE_CODES.get(symptom, ""),
                "symptom": pretty_symptom(symptom),
                "pred_column": symptom,
                "cluster": network.partition[symptom],
                "prevalence_pct": round(prevalence[symptom] * 100, 1),
                "n_patients": int(network.matrix[symptom].sum()),
                "strength_centrality": round(float(centrality[symptom]), 2),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["cluster", "strength_centrality"], ascending=[True, False]
    ).reset_index(drop=True)

    if paths is not None:
        table.to_csv(
            paths.ensure_results() / "table4_symptom_clusters.csv", index=False
        )
    return table
