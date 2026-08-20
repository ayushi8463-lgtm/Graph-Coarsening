"""
Interactive Graph Builder + UGC Coarsening GUI (pure matplotlib)
===================================================================

HOW TO USE
----------
    pip install matplotlib networkx numpy
    python graph_builder_gui.py

MODES (click the buttons at the bottom, or use keys):
  [A] Add Node mode   - click anywhere on the canvas to drop a new node
  [E] Add Edge mode   - click one node, then click another, to connect them
  [D] Delete mode     - click a node to delete it (and its edges),
                        or click near the midpoint of an edge to delete it
  [C] Coarsen         - run UGC coarsening on your current graph and show
                        the result in the right panel
  [R] Reset           - clear everything and start over

Node features (needed for UGC's augmented-feature step) are auto-generated
as random 3D vectors when a node is created -- shown as small numbers you
don't need to worry about; the point is every node still gets a feature
vector like in the earlier scripts.

The bin-width slider controls how aggressively UGC coarsens once you hit
[C]. Nodes in the same coarsened super-node share a colour on BOTH panels.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons

np.random.seed(0)

# ----------------------------------------------------------------------
# Graph state (plain python, no networkx needed for the editable graph)
# ----------------------------------------------------------------------
node_positions = {}     # node_id -> (x, y)
node_features = {}      # node_id -> np.array of 3 random features
edges = set()            # set of frozenset({i, j})
next_node_id = [0]       # mutable counter

mode = ["add_node"]      # current interaction mode
selected_node = [None]    # for edge-adding: first clicked node waiting for second click

# coarsening result cache (populated when [C] is pressed)
coarsen_result = {"groups": None, "Ac": None, "Fc": None}


def add_node(x, y):
    nid = next_node_id[0]
    node_positions[nid] = (x, y)
    node_features[nid] = np.random.randn(3) * 0.5   # random toy feature vector
    next_node_id[0] += 1
    return nid


def add_edge(i, j):
    if i != j:
        edges.add(frozenset((i, j)))


def delete_node(nid):
    if nid in node_positions:
        del node_positions[nid]
        del node_features[nid]
        edges_to_remove = [e for e in edges if nid in e]
        for e in edges_to_remove:
            edges.discard(e)


def nearest_node(x, y, threshold=0.06):
    if not node_positions:
        return None
    dists = {nid: (px - x) ** 2 + (py - y) ** 2
             for nid, (px, py) in node_positions.items()}
    nid = min(dists, key=dists.get)
    return nid if dists[nid] ** 0.5 < threshold else None


def nearest_edge(x, y, threshold=0.05):
    best, best_d = None, threshold
    for e in edges:
        i, j = tuple(e)
        xi, yi = node_positions[i]
        xj, yj = node_positions[j]
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        d = ((mx - x) ** 2 + (my - y) ** 2) ** 0.5
        if d < best_d:
            best, best_d = e, d
    return best


# ----------------------------------------------------------------------
# UGC coarsening (same core pipeline as earlier scripts)
# ----------------------------------------------------------------------
def run_ugc(r, alpha=0.4, l_proj=4, seed=1):
    ids = sorted(node_positions.keys())
    N = len(ids)
    if N == 0:
        return [], np.zeros((0, 0)), np.zeros((0, 3))
    id_to_idx = {nid: k for k, nid in enumerate(ids)}

    A = np.zeros((N, N))
    for e in edges:
        i, j = tuple(e)
        if i in id_to_idx and j in id_to_idx:
            A[id_to_idx[i], id_to_idx[j]] = 1
            A[id_to_idx[j], id_to_idx[i]] = 1

    X = np.array([node_features[nid] for nid in ids])
    F = np.concatenate([(1 - alpha) * X, alpha * A], axis=1)

    rng = np.random.RandomState(seed)
    d = F.shape[1]
    W = rng.randn(d, l_proj)
    b = rng.uniform(0, 1.0, size=l_proj)

    H = np.floor((F @ W + b) / r)
    hash_values = np.array([
        np.unique(H[i], return_counts=True)[0][
            np.argmax(np.unique(H[i], return_counts=True)[1])
        ] for i in range(N)
    ])

    unique_hashes = np.unique(hash_values)
    hash_to_super = {h: s for s, h in enumerate(unique_hashes)}
    n_super = len(unique_hashes)

    C = np.zeros((N, n_super))
    for i in range(N):
        C[i, hash_to_super[hash_values[i]]] = 1

    sizes = C.sum(axis=0)
    Ac = C.T @ A @ C
    Fc = (C.T @ X) / sizes[:, None]

    # groups expressed in ORIGINAL node ids (not matrix indices)
    groups = [[ids[i] for i in range(N) if hash_values[i] == h] for h in unique_hashes]
    return groups, Ac, Fc


# ----------------------------------------------------------------------
# FIGURE + AXES layout
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(13, 7))
ax_left = fig.add_axes([0.05, 0.28, 0.42, 0.62])
ax_right = fig.add_axes([0.53, 0.28, 0.42, 0.62])
ax_left.set_xlim(0, 1); ax_left.set_ylim(0, 1)
ax_left.set_title("Click to build your graph", fontsize=11)
ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)
ax_right.set_title("UGC-coarsened result (press Coarsen)", fontsize=11)
ax_left.set_xticks([]); ax_left.set_yticks([])
ax_right.set_xticks([]); ax_right.set_yticks([])

cmap = plt.colormaps.get_cmap("tab10")

status_text = fig.text(0.05, 0.94, "", fontsize=11, fontweight="bold", color="darkblue")


def set_status(msg):
    status_text.set_text(msg)
    fig.canvas.draw_idle()


# ----------------------------------------------------------------------
# DRAW functions
# ----------------------------------------------------------------------
def node_color_map():
    """If a coarsening result exists, colour nodes by super-node; else all grey."""
    groups = coarsen_result["groups"]
    colors = {}
    if groups is None:
        for nid in node_positions:
            colors[nid] = "lightgrey"
    else:
        for s_idx, members in enumerate(groups):
            c = cmap(s_idx % 10)
            for m in members:
                colors[m] = c
    return colors


def draw_left():
    ax_left.clear()
    ax_left.set_xlim(0, 1); ax_left.set_ylim(0, 1)
    ax_left.set_xticks([]); ax_left.set_yticks([])
    ax_left.set_title(f"Your graph  ({len(node_positions)} nodes, {len(edges)} edges)",
                       fontsize=11)

    colors = node_color_map()

    for e in edges:
        i, j = tuple(e)
        if i in node_positions and j in node_positions:
            xi, yi = node_positions[i]
            xj, yj = node_positions[j]
            ax_left.plot([xi, xj], [yi, yj], color="black", lw=1.5, zorder=1)

    for nid, (x, y) in node_positions.items():
        c = colors.get(nid, "lightgrey")
        edgecolor = "red" if nid == selected_node[0] else "black"
        lw = 2.5 if nid == selected_node[0] else 1
        ax_left.scatter([x], [y], s=500, color=[c], edgecolors=edgecolor,
                         linewidths=lw, zorder=2)
        ax_left.text(x, y, str(nid), ha="center", va="center",
                      fontsize=9, fontweight="bold", zorder=3)

    fig.canvas.draw_idle()


def draw_right():
    ax_right.clear()
    ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)
    ax_right.set_xticks([]); ax_right.set_yticks([])

    groups, Ac, Fc = coarsen_result["groups"], coarsen_result["Ac"], coarsen_result["Fc"]
    if groups is None or len(groups) == 0:
        ax_right.set_title("UGC-coarsened result (press Coarsen)", fontsize=11)
        fig.canvas.draw_idle()
        return

    n_super = len(groups)
    ax_right.set_title(
        f"Coarsened: {len(node_positions)} -> {n_super} super-nodes "
        f"({(1 - n_super/max(1,len(node_positions))):.0%} reduction)",
        fontsize=11
    )

    # simple circular layout for the coarsened graph
    angles = np.linspace(0, 2 * np.pi, n_super, endpoint=False)
    pos_c = {s: (0.5 + 0.35 * np.cos(a), 0.5 + 0.35 * np.sin(a))
              for s, a in enumerate(angles)}

    max_w = Ac.max() if Ac.size and Ac.max() > 0 else 1
    for i in range(n_super):
        for j in range(i + 1, n_super):
            w = Ac[i, j]
            if w > 0:
                xi, yi = pos_c[i]
                xj, yj = pos_c[j]
                ax_right.plot([xi, xj], [yi, yj], color="black",
                              lw=1 + 3 * (w / max_w), zorder=1)

    sizes_list = [len(g) for g in groups]
    for s_idx in range(n_super):
        x, y = pos_c[s_idx]
        c = cmap(s_idx % 10)
        size = 400 + 250 * sizes_list[s_idx]
        # self-loop weight (internal edges) shown as a ring thickness proxy
        ax_right.scatter([x], [y], s=size, color=[c], edgecolors="black",
                          linewidths=1.5, zorder=2)
        ax_right.text(x, y, f"S{s_idx}\n({sizes_list[s_idx]})",
                      ha="center", va="center", fontsize=8,
                      fontweight="bold", zorder=3)

    fig.canvas.draw_idle()


def redraw():
    draw_left()
    draw_right()


# ----------------------------------------------------------------------
# EVENT HANDLING
# ----------------------------------------------------------------------
def on_click(event):
    if event.inaxes != ax_left:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return

    if mode[0] == "add_node":
        add_node(x, y)
        coarsen_result["groups"] = None  # invalidate stale coarsening
        set_status(f"Added node {next_node_id[0]-1}. Mode: Add Node")

    elif mode[0] == "add_edge":
        nid = nearest_node(x, y)
        if nid is None:
            set_status("Click closer to a node to select it for an edge.")
            return
        if selected_node[0] is None:
            selected_node[0] = nid
            set_status(f"Selected node {nid}. Click another node to connect.")
        else:
            if nid != selected_node[0]:
                add_edge(selected_node[0], nid)
                set_status(f"Connected {selected_node[0]} -- {nid}. Mode: Add Edge")
            coarsen_result["groups"] = None
            selected_node[0] = None

    elif mode[0] == "delete":
        nid = nearest_node(x, y)
        if nid is not None:
            delete_node(nid)
            set_status(f"Deleted node {nid}.")
        else:
            e = nearest_edge(x, y)
            if e is not None:
                edges.discard(e)
                set_status("Deleted an edge.")
            else:
                set_status("Nothing close enough to delete.")
        coarsen_result["groups"] = None

    redraw()


fig.canvas.mpl_connect("button_press_event", on_click)


# ----------------------------------------------------------------------
# MODE BUTTONS (radio)
# ----------------------------------------------------------------------
ax_radio = fig.add_axes([0.05, 0.05, 0.18, 0.15])
radio = RadioButtons(ax_radio, ["Add Node", "Add Edge", "Delete"])


def on_mode_change(label):
    mapping = {"Add Node": "add_node", "Add Edge": "add_edge", "Delete": "delete"}
    mode[0] = mapping[label]
    selected_node[0] = None
    set_status(f"Mode: {label}")
    redraw()


radio.on_clicked(on_mode_change)

# ----------------------------------------------------------------------
# BIN-WIDTH SLIDER (controls UGC coarsening aggressiveness)
# ----------------------------------------------------------------------
ax_slider = fig.add_axes([0.30, 0.14, 0.35, 0.03])
r_slider = Slider(ax_slider, "bin-width r", 0.1, 2.0, valinit=0.6, valstep=0.05)

# ----------------------------------------------------------------------
# COARSEN BUTTON
# ----------------------------------------------------------------------
ax_coarsen_btn = fig.add_axes([0.30, 0.05, 0.15, 0.06])
btn_coarsen = Button(ax_coarsen_btn, "Coarsen (C)")


def on_coarsen(event):
    if len(node_positions) == 0:
        set_status("Add some nodes first!")
        return
    groups, Ac, Fc = run_ugc(r_slider.val)
    coarsen_result["groups"] = groups
    coarsen_result["Ac"] = Ac
    coarsen_result["Fc"] = Fc
    set_status(f"Coarsened at r={r_slider.val:.2f}: "
               f"{len(node_positions)} -> {len(groups)} super-nodes")
    redraw()


btn_coarsen.on_clicked(on_coarsen)


def on_slider_change(val):
    # if a coarsening already exists, keep it live-updating as slider moves
    if coarsen_result["groups"] is not None:
        on_coarsen(None)


r_slider.on_changed(on_slider_change)

# ----------------------------------------------------------------------
# RESET BUTTON
# ----------------------------------------------------------------------
ax_reset_btn = fig.add_axes([0.50, 0.05, 0.15, 0.06])
btn_reset = Button(ax_reset_btn, "Reset (R)")


def on_reset(event):
    node_positions.clear()
    node_features.clear()
    edges.clear()
    next_node_id[0] = 0
    selected_node[0] = None
    coarsen_result["groups"] = None
    set_status("Cleared. Mode: " + radio.value_selected)
    redraw()


btn_reset.on_clicked(on_reset)


# ----------------------------------------------------------------------
# KEYBOARD SHORTCUTS
# ----------------------------------------------------------------------
def on_key(event):
    if event.key == "a":
        radio.set_active(0)
    elif event.key == "e":
        radio.set_active(1)
    elif event.key == "d":
        radio.set_active(2)
    elif event.key == "c":
        on_coarsen(None)
    elif event.key == "r":
        on_reset(None)


fig.canvas.mpl_connect("key_press_event", on_key)

set_status("Mode: Add Node -- click on the left panel to place nodes")
redraw()
plt.show()
