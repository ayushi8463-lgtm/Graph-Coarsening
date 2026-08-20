"""
Interactive AH-UGC Demo: Adaptive Graph Coarsening
=====================================================

THE KEY IDEA THIS DEMO PROVES
-------------------------------
AH-UGC computes hashing + sorting ONCE, then merges neighbors one pair at a
time until only 1 super-node is left. This full "merge history" is computed
a SINGLE time, up front.

The slider below lets you scrub through EVERY coarsening ratio from 100%
down to 10%. Moving the slider does NOT re-hash or re-sort anything -- it
just looks up which point in the already-computed merge history matches
your chosen ratio. That's the "adaptive" property in action: infinite
resolutions, one computation.

(Compare this to plain UGC, where changing the ratio means picking a new
bin-width r and re-running the whole hashing pipeline from scratch.)

LEFT PANEL:  original 10-node graph, nodes coloured by current super-node
RIGHT PANEL: the coarsened graph at the slider's ratio

HOW TO RUN
----------
    pip install matplotlib networkx numpy
    python ah_ugc_interactive.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

np.random.seed(42)

# ----------------------------------------------------------------------
# STEP 0: Same fixed 10-node toy graph as before
# ----------------------------------------------------------------------
N = 10
edges = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),   # cluster A: 0,1,2,3
    (4, 5), (4, 6), (5, 6), (5, 7), (6, 7),   # cluster B: 4,5,6,7
    (8, 9),                                    # cluster C: 8,9
    (3, 4), (7, 8),                            # bridges
]
A = np.zeros((N, N))
for i, j in edges:
    A[i, j] = A[j, i] = 1

X = np.array([
    [1.0, 0.1, 0.0], [1.1, 0.0, 0.1], [0.9, 0.2, -0.1], [1.0, -0.1, 0.05],
    [-1.0, 1.0, 0.2], [-0.9, 1.1, 0.1], [-1.1, 0.9, -0.1], [-1.0, 1.0, 0.3],
    [0.1, -1.0, 1.0], [0.0, -1.1, 0.9],
])

alpha = 0.4
F = np.concatenate([(1 - alpha) * X, alpha * A], axis=1)   # (10, 13) augmented features

# ----------------------------------------------------------------------
# STEP 1-3: LSH projections -> aggregate -> sort (all done ONCE)
# ----------------------------------------------------------------------
l_proj = 4
d = F.shape[1]
W = np.random.randn(d, l_proj)
b = np.random.uniform(0, 1.0, size=l_proj)

S = F @ W + b                # raw projections, (10, 4)
s = S.mean(axis=1)           # aggregated scalar score per node, (10,)
order = np.argsort(s)        # sorted node order = the "hashing ring"

print("Sorted node order (consistent hashing ring):", list(order))
print("Scores:", {int(i): round(float(s[i]), 3) for i in order})


# ----------------------------------------------------------------------
# STEP 4: Build the FULL merge history ONCE, from N super-nodes down to 1.
#          This is the entire cost AH-UGC ever pays -- the slider below
#          reuses this list with ZERO extra hashing/sorting.
# ----------------------------------------------------------------------
def build_merge_history(order, rng):
    """
    Returns a list `history` where history[k] = list of groups when there
    are (N - k) super-nodes remaining, i.e. history[0] is fully unmerged
    (N singleton groups), history[N-1] is the single all-in-one group.
    """
    current = [[int(node)] for node in order]
    history = [ [g[:] for g in current] ]  # snapshot at N groups
    while len(current) > 1:
        k = len(current)
        j = rng.integers(0, k)
        right = (j + 1) % k
        merged = current[j] + current[right]
        current = [merged if idx == j else current[idx]
                   for idx in range(k) if idx != right]
        history.append([g[:] for g in current])
    return history  # history[i] has (N - i) groups


rng = np.random.default_rng(7)
merge_history = build_merge_history(order, rng)
# merge_history[i] -> N - i super-nodes exist at this point
group_counts = [len(h) for h in merge_history]
print(f"\nMerge history computed ONCE: {len(merge_history)} snapshots, "
      f"from {group_counts[0]} super-nodes down to {group_counts[-1]}.")


def get_groups_for_ratio(ratio):
    """
    Look up (NOT recompute) the merge-history snapshot closest to the
    requested coarsening ratio. This is an O(1) list lookup -- the whole
    point of AH-UGC's adaptivity.
    """
    target_count = max(1, round(N * ratio))
    # find the history snapshot with a super-node count closest to target
    best_idx = min(range(len(merge_history)),
                    key=lambda i: abs(group_counts[i] - target_count))
    return merge_history[best_idx]


# ----------------------------------------------------------------------
# STEP 5: Build coarsened graph (Ac, Fc) for a given grouping -- same
#          CᵀAC / average construction as UGC, nothing new here.
# ----------------------------------------------------------------------
def build_coarsened_graph(groups):
    n_super = len(groups)
    C = np.zeros((N, n_super))
    for s_idx, members in enumerate(groups):
        for m in members:
            C[m, s_idx] = 1
    sizes = C.sum(axis=0)
    Ac = C.T @ A @ C
    Fc = (C.T @ F) / sizes[:, None]
    return C, Ac, Fc


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------
G_orig = nx.from_numpy_array(A)
pos_orig = nx.spring_layout(G_orig, seed=7)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6))
plt.subplots_adjust(bottom=0.22)

cmap = plt.colormaps.get_cmap("tab10")


def draw(ratio):
    ax_left.clear()
    ax_right.clear()

    groups = get_groups_for_ratio(ratio)
    n_super = len(groups)
    C, Ac, Fc = build_coarsened_graph(groups)

    super_colors = {s: cmap(s % 10) for s in range(n_super)}
    node_colors = [None] * N
    for s_idx, members in enumerate(groups):
        for m in members:
            node_colors[m] = super_colors[s_idx]

    # ---- LEFT: original graph ----
    nx.draw(
        G_orig, pos_orig, ax=ax_left,
        with_labels=True, node_color=node_colors,
        edgecolors="black", node_size=650, font_weight="bold"
    )
    ax_left.set_title(
        f"Original graph (N={N})\ncolour = current super-node membership",
        fontsize=10
    )

    # ---- RIGHT: coarsened graph ----
    G_c = nx.from_numpy_array(Ac)
    pos_c = nx.spring_layout(G_c, seed=3)

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
    if edge_weights:
        nx.draw_networkx_edges(
            G_c, pos_c, ax=ax_right,
            width=[1 + 3 * (w / max_w) for w in edge_weights]
        )

    actual_ratio = n_super / N
    ax_right.set_title(
        f"Coarsened graph: {N} -> {n_super} super-nodes "
        f"(actual ratio = {actual_ratio:.0%})\n"
        f"looked up instantly from the SAME precomputed merge history",
        fontsize=10
    )

    ax_left.axis("off")
    ax_right.axis("off")
    fig.canvas.draw_idle()


# ----------------------------------------------------------------------
# SLIDER: target coarsening ratio (fraction of nodes remaining)
# ----------------------------------------------------------------------
ax_slider = plt.axes([0.25, 0.06, 0.5, 0.04])
ratio_slider = Slider(ax_slider, "target ratio (fraction remaining)",
                       0.1, 1.0, valinit=1.0, valstep=0.05)


def on_change(val):
    draw(ratio_slider.val)


ratio_slider.on_changed(on_change)

fig.suptitle(
    "AH-UGC: one merge pass computed ONCE -- slider scrubs through every "
    "ratio with zero recomputation",
    fontsize=11, fontweight="bold"
)

draw(ratio_slider.val)
plt.show()