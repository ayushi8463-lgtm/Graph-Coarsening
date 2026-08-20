"""
Interactive UGC-style Graph Coarsening Demo
=============================================

WHAT THIS DOES
---------------
Shows a 10-node graph on the LEFT, and its coarsened version on the RIGHT.
A slider at the bottom controls the LSH bin-width "r". As you drag it,
the coarsening is recomputed live and both panels update.

- Small r  -> hash buckets are narrow -> fewer nodes collide -> LESS coarsening
             (more super-nodes, closer to the original graph)
- Large r  -> hash buckets are wide  -> more nodes collide -> MORE coarsening
             (fewer super-nodes, more aggressive compression)

Nodes belonging to the same super-node are drawn in the same colour on
BOTH panels, so you can visually track which original nodes got merged.

HOW TO RUN
----------
    pip install matplotlib networkx numpy
    python ugc_interactive.py

Then just drag the "bin-width r" slider.

THE ALGORITHM (this is the real UGC pipeline, not a simplification)
---------------------------------------------------------------------
1. Build augmented feature F_i = concat( (1-alpha)*X_i , alpha*A_i )
   -> blends node features (X) with structural/adjacency info (A),
      weighted by a heterophily factor alpha.

2. Pick l random projection vectors w_1..w_l (columns of W), each drawn
   from a Gaussian (a 2-stable distribution -- required for the LSH
   collision-probability guarantees UGC proves in the paper).

3. For each node i and projector k:
       h_i^k = floor( (F_i . w_k + b_k) / r )
   This is standard "stable-distribution LSH": project onto a random
   line, add a random offset b_k, then quantize into bins of width r.
   Nearby points in F-space are likely to land in the same bin.

4. Each node's FINAL hash value = the most frequently occurring h_i^k
   across all l projectors (majority vote -> more robust than a single
   hash function).

5. All nodes sharing the same final hash value become ONE super-node.
   This directly builds the coarsening matrix C (N x n), where
   C[i, s] = 1 iff node i belongs to super-node s.

6. Coarsened graph:
       A_c = C^T A C      (edge weight between super-nodes = total
                            number of original edges crossing between
                            their member sets; diagonal = internal edges)
       F_c = (C^T F) / (size of each super-node)   (average features)

This is EXACTLY the construction described in Algorithm 1 of the UGC
paper (Kataria, Kumar, Jayadeva - NeurIPS 2024), just on a tiny toy
graph so you can see every step.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ----------------------------------------------------------------------
# STEP 0: Fixed toy graph + features (does not change as you move slider)
# ----------------------------------------------------------------------
np.random.seed(42)          # graph/feature randomness fixed
N = 10

edges = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),   # cluster A: 0,1,2,3
    (4, 5), (4, 6), (5, 6), (5, 7), (6, 7),   # cluster B: 4,5,6,7
    (8, 9),                                    # cluster C: 8,9
    (3, 4), (7, 8),                            # bridges between clusters
]

A = np.zeros((N, N))
for i, j in edges:
    A[i, j] = A[j, i] = 1

# Node features: three "centroids" (one per visual cluster) + small noise
X = np.array([
    [1.0, 0.1, 0.0],
    [1.1, 0.0, 0.1],
    [0.9, 0.2, -0.1],
    [1.0, -0.1, 0.05],
    [-1.0, 1.0, 0.2],
    [-0.9, 1.1, 0.1],
    [-1.1, 0.9, -0.1],
    [-1.0, 1.0, 0.3],
    [0.1, -1.0, 1.0],
    [0.0, -1.1, 0.9],
])

alpha = 0.4          # heterophily-weighting factor between features & structure
l_proj = 4            # number of LSH hash functions (projectors)

# Random projectors are also fixed via the seed, so ONLY r changes as you
# move the slider -- this isolates the effect of bin-width cleanly.
d = X.shape[1] + N          # augmented feature dimension = feat dims + N (adjacency row)
W = np.random.randn(d, l_proj)          # Gaussian projectors (2-stable dist)
b = np.random.uniform(0, 1.0, size=l_proj)  # fixed per-projector offsets

# Augmented feature matrix F (built once; independent of r)
F = np.concatenate([(1 - alpha) * X, alpha * A], axis=1)

# Fixed layout for the ORIGINAL graph, reused every redraw for stability
G_orig = nx.from_numpy_array(A)
pos_orig = nx.spring_layout(G_orig, seed=7)


# ----------------------------------------------------------------------
# CORE FUNCTION: run the UGC coarsening pipeline for a given bin-width r
# ----------------------------------------------------------------------
def coarsen(r):
    """
    Given bin-width r, returns:
      hash_values : (N,) final hash id per node
      C           : (N, n_super) coarsening matrix
      Ac          : (n_super, n_super) coarsened adjacency
      Fc          : (n_super, d) coarsened (averaged) features
      groups      : dict {super_node_index: [original node ids]}
    """
    # per-node, per-projector hash index: floor((F.w + b) / r)
    H = np.floor((F @ W + b) / r)          # shape (N, l_proj)

    # majority-vote across the l_proj hash functions -> final hash per node
    hash_values = np.array([
        np.unique(H[i], return_counts=True)[0][
            np.argmax(np.unique(H[i], return_counts=True)[1])
        ]
        for i in range(N)
    ])

    unique_hashes = np.unique(hash_values)
    n_super = len(unique_hashes)
    hash_to_super = {h: s for s, h in enumerate(unique_hashes)}

    # Build coarsening matrix C (N x n_super)
    C = np.zeros((N, n_super))
    for i in range(N):
        C[i, hash_to_super[hash_values[i]]] = 1

    sizes = C.sum(axis=0)
    Ac = C.T @ A @ C
    Fc = (C.T @ F) / sizes[:, None]

    groups = {s: [i for i in range(N) if hash_to_super[hash_values[i]] == s]
              for s in range(n_super)}

    return hash_values, C, Ac, Fc, groups


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6))
plt.subplots_adjust(bottom=0.22)

cmap = plt.colormaps.get_cmap("tab10")


def draw(r):
    ax_left.clear()
    ax_right.clear()

    hash_values, C, Ac, Fc, groups = coarsen(r)
    n_super = len(groups)

    # assign each super-node a colour, and map back to per-node colours
    super_colors = {s: cmap(s % 10) for s in groups}
    node_colors = [None] * N
    for s, members in groups.items():
        for m in members:
            node_colors[m] = super_colors[s]

    # ---- LEFT PANEL: original graph, nodes coloured by their super-node ----
    nx.draw(
        G_orig, pos_orig, ax=ax_left,
        with_labels=True, node_color=node_colors,
        edgecolors="black", node_size=650, font_weight="bold"
    )
    ax_left.set_title(
        f"Original graph (N={N} nodes)\ncolour = which super-node it will join",
        fontsize=10
    )

    # ---- RIGHT PANEL: coarsened graph ----
    G_c = nx.from_numpy_array(Ac)
    pos_c = nx.spring_layout(G_c, seed=3)  # fresh layout since #nodes changes

    # node size proportional to how many original nodes it absorbed
    sizes_list = [len(groups[s]) for s in range(n_super)]
    node_sizes = [400 + 250 * sz for sz in sizes_list]
    colors_c = [super_colors[s] for s in range(n_super)]

    edge_weights = [G_c[u][v]['weight'] for u, v in G_c.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_nodes(G_c, pos_c, ax=ax_right, node_color=colors_c,
                            node_size=node_sizes, edgecolors="black")
    nx.draw_networkx_labels(
        G_c, pos_c, ax=ax_right,
        labels={s: f"S{s}\n({sizes_list[s]})" for s in range(n_super)},
        font_size=8, font_weight="bold"
    )
    nx.draw_networkx_edges(
        G_c, pos_c, ax=ax_right,
        width=[1 + 3 * (w / max_w) for w in edge_weights]
    )

    ratio = 1 - n_super / N
    ax_right.set_title(
        f"Coarsened graph: {N} -> {n_super} super-nodes "
        f"(coarsening ratio = {ratio:.0%})\n"
        f"edge thickness = merged-edge weight, node size = #members",
        fontsize=10
    )

    ax_left.axis("off")
    ax_right.axis("off")
    fig.canvas.draw_idle()


# ----------------------------------------------------------------------
# SLIDER: bin-width r
# ----------------------------------------------------------------------
ax_slider = plt.axes([0.25, 0.06, 0.5, 0.04])
r_slider = Slider(ax_slider, "bin-width r", 0.10, 2.0, valinit=0.35, valstep=0.01)


def on_change(val):
    draw(r_slider.val)


r_slider.on_changed(on_change)

draw(r_slider.val)  # initial render
plt.show()
