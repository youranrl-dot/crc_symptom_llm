"""Figure generation.

Figure 3 is the symptom phi correlation matrix. Two choices are worth flagging:

* A **sequential** single-hue ramp, not a diverging red/blue scale. No negative
  phi correlation survives the prevalence filter, so a diverging scale would
  imply a polarity the data do not contain.
* The **diagonal is masked**. Self-correlations are 1.0 by construction and would
  otherwise dominate the colour scale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .config import (  # noqa: E402
    CLUSTER_LABELS,
    DIAGONAL_FILL,
    INK,
    MUTED,
    NODE_CODES,
    PUBLISHED_CLUSTERS,
    SEQUENTIAL_BLUE,
    SURFACE,
    Paths,
    pretty_symptom,
)
from .network import SymptomNetwork  # noqa: E402

__all__ = ["plot_phi_matrix"]

_CLUSTER_TITLES = {
    "Systemic": "Cluster 1\nSystemic",
    "CRC": "Cluster 2\nCRC disease-specific",
    "GI": "Cluster 3\nGastrointestinal",
}


def _ordered_symptoms(network: SymptomNetwork) -> tuple[list[str], list[tuple[int, int, str]]]:
    """Order rows/columns by published cluster, then by strength centrality."""
    centrality = network.strength_centrality()
    ordered: list[str] = []
    groups: list[tuple[int, int, str]] = []
    for name, members in PUBLISHED_CLUSTERS.items():
        present = [s for s in members if s in network.retained]
        present.sort(key=lambda s: float(centrality[s]), reverse=True)
        start = len(ordered)
        ordered.extend(present)
        groups.append((start, len(ordered), _CLUSTER_TITLES[name]))
    # Any retained symptom not in the published clusters is appended so the
    # figure never silently drops a node.
    extra = [s for s in network.retained if s not in ordered]
    if extra:
        start = len(ordered)
        ordered.extend(extra)
        groups.append((start, len(ordered), "Unassigned"))
    return ordered, groups


def plot_phi_matrix(
    network: SymptomNetwork,
    out_path: Path | str | None = None,
    *,
    paths: Paths | None = None,
    vmax: float = 0.80,
    dpi: int = 300,
) -> Path:
    """Render Figure 3 and return the path written."""
    if out_path is None:
        base = (paths or Paths()).ensure_results()
        out_path = Path(base) / "figure3_phi_matrix.png"
    out_path = Path(out_path)

    order, groups = _ordered_symptoms(network)
    k = len(order)
    phi = network.phi.loc[order, order].values
    y_labels = [f"{NODE_CODES.get(s, '')}   {pretty_symptom(s)}".strip() for s in order]
    x_labels = [NODE_CODES.get(s, pretty_symptom(s)) for s in order]
    boundaries = [end for _, end, _ in groups[:-1]]

    cmap = LinearSegmentedColormap.from_list("seq_blue", list(SEQUENTIAL_BLUE), N=256)
    norm = Normalize(0, vmax)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": dpi})

    # Explicit inch-based layout: the axes is square, so every margin is known in
    # advance and no label can collide with another element.
    fig_w, fig_h = 7.30, 5.95
    left, right, top = 1.86, 0.92, 1.05
    axes_w = fig_w - left - right
    axes_h = axes_w

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, (fig_h - top - axes_h) / fig_h,
                       axes_w / fig_w, axes_h / fig_h])

    masked = np.ma.masked_array(phi, mask=np.eye(k, dtype=bool))
    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    for i in range(k):
        ax.add_patch(
            Rectangle((i - .5, i - .5), 1, 1, facecolor=DIAGONAL_FILL,
                      edgecolor=SURFACE, lw=.6, zorder=2)
        )

    threshold = network.phi_threshold
    for i in range(k):
        for j in range(k):
            if i != j and phi[i, j] >= threshold:
                ax.text(
                    j, i, f"{phi[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=5.0, zorder=3,
                    color=SURFACE if phi[i, j] >= vmax * 0.525 else INK,
                )

    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, k, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=.55)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(x_labels, fontsize=6.4, color=INK)
    ax.set_yticklabels(y_labels, fontsize=6.6, color=INK)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for b in boundaries:
        ax.axhline(b - .5, color=INK, lw=1.2, zorder=4)
        ax.axvline(b - .5, color=INK, lw=1.2, zorder=4)
    ax.add_patch(Rectangle((-.5, -.5), k, k, fill=False, edgecolor=INK, lw=1.2, zorder=4))

    band_y, band_h = -1.90, 0.95
    for start, end, title in groups:
        ax.add_patch(
            Rectangle((start - .42, band_y), (end - start) - .16, band_h,
                      facecolor="#eceae5", edgecolor="none", clip_on=False, zorder=5)
        )
        ax.text((start + end - 1) / 2, band_y + band_h / 2, title, ha="center",
                va="center", fontsize=6.0, linespacing=1.4, color=MUTED,
                clip_on=False, zorder=6)

    cax = fig.add_axes([(left + axes_w + 0.16) / fig_w,
                        (fig_h - top - axes_h) / fig_h, 0.017, axes_h / fig_h])
    bar = fig.colorbar(image, cax=cax, ticks=[0, .2, .4, .6, .8])
    bar.set_label("Phi correlation (φ)", fontsize=7.2, color=INK, labelpad=5)
    bar.ax.tick_params(labelsize=6.6, length=2, color=MUTED, labelcolor=INK)
    bar.outline.set_visible(False)

    fig.text(left / fig_w, (fig_h - 0.42) / fig_h,
             f"Symptom phi correlation matrix — Gemini 3.5 Flash "
             f"(N = {network.n_patients:,})",
             fontsize=9.2, color=INK, ha="left", va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out_path
