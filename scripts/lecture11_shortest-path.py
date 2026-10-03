# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# description: "The shortest path problem: an LO model as a special transshipment problem, and Dijkstra's algorithm step by step."
# thumbnail: null
# ---
# # Lecture 11: Shortest Path
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture11_shortest-path.ipynb)

# %% [markdown]
# This lecture looks at three classical problems: the shortest path problem in this
# notebook, then [maximum flow](lecture11_maximum-flow.ipynb) and the
# [traveling salesman problem](lecture11_tsp.ipynb). Each notebook takes the same
# steps: we define the problem and give an example, solve the example with an LO
# approach (a model solved with pulp, as in Lectures 8 to 10), and then solve it with
# a dedicated algorithm. The dedicated algorithm uses the structure of the problem and
# is much faster, so it is often the way to go in practice (see
# [Algorithms and Their Characteristics](lecture11_algorithms.ipynb)). For the
# shortest path problem, that algorithm is Dijkstra's algorithm.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate the shortest path problem as an LO model and solve it with pulp;
# - explain what $d_W(x)$ means at every step of Dijkstra's algorithm;
# - apply Dijkstra's algorithm by hand in a table and find a shortest path by
#   backtracking;
# - explain why Dijkstra's algorithm needs non-negative distances and why its running
#   time grows like $n^2$.

# %%
import math

import matplotlib.pyplot as plt
import pandas as pd
import pulp
from matplotlib.patches import Circle, FancyArrowPatch

# %% tags=["hide-input"]
# colors that read on the light and the dark site theme
NODE_FILL, NODE_EDGE = "#d6dce5", "#5b9bd5"
EDGE_COLOR, TEXT_COLOR = "#8c8c8c", "#111827"
NODE_RADIUS = 0.2


def draw_graph(
    positions,
    arcs,
    highlight=(),
    highlight_color="#d62728",
    directed=True,
    both_ways=(),
    label_at=None,
    node_notes=None,
    cut_x=None,
    figsize=(6, 3.2),
):
    """Draw a graph with a number (label) on every arc.

    positions: node -> (x, y)
    arcs: (i, j) -> label; both_ways: arcs drawn with an arrow at both ends
    highlight: arcs drawn thick in highlight_color
    label_at: (i, j) -> position of the label along the arc (default 0.5)
    node_notes: node -> (text, dx, dy) for a box next to the node
    cut_x: x-coordinate of a vertical line that marks a cut

    The background is transparent and every text sits on its own box, so
    the figure reads the same in the light and the dark site theme.
    """
    label_at = label_at or {}
    node_notes = node_notes or {}
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    for (i, j), label in arcs.items():
        (x1, y1), (x2, y2) = positions[i], positions[j]
        on = (i, j) in highlight or (not directed and (j, i) in highlight)
        color = highlight_color if on else EDGE_COLOR
        style = "-"
        if directed:
            style = "<|-|>" if (i, j) in both_ways else "-|>"
        # start and end the arc at the border of the node circles
        length = math.hypot(x2 - x1, y2 - y1)
        dx = (x2 - x1) / length * NODE_RADIUS
        dy = (y2 - y1) / length * NODE_RADIUS
        arrow = FancyArrowPatch(
            (x1 + dx, y1 + dy),
            (x2 - dx, y2 - dy),
            arrowstyle=style,
            mutation_scale=14,
            color=color,
            lw=3 if on else 1.5,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)
        t = label_at.get((i, j), 0.5)
        ax.text(
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1),
            str(label),
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT_COLOR,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color},
        )
    for node, (x, y) in positions.items():
        circle = Circle(
            (x, y), NODE_RADIUS, fc=NODE_FILL, ec=NODE_EDGE, lw=1.5, zorder=3
        )
        ax.add_patch(circle)
        ax.text(x, y, node, ha="center", va="center", fontsize=12, zorder=4)
    for node, (text, dx, dy) in node_notes.items():
        x, y = positions[node]
        ax.text(
            x + dx,
            y + dy,
            text,
            ha="center",
            va="center",
            fontsize=11,
            color=highlight_color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": highlight_color,
            },
        )
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    if cut_x is not None:
        ax.plot(
            [cut_x, cut_x],
            [min(ys) - 0.4, max(ys) + 0.4],
            color="#d62728",
            lw=3,
        )
    ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
    ax.set_ylim(min(ys) - 0.45, max(ys) + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()


# %% [markdown]
# ## The Shortest Path Problem
#
# Every time you ask navigation software for a route, it solves a shortest path
# problem: find the shortest (or fastest) route from where you are to where you want
# to go. The same problem appears in many less visible places, for example in routing
# data packets through the internet, and as a building block of other problems: the
# [traveling salesman problem](lecture11_tsp.ipynb) needs the shortest
# distance between every pair of cities.
#
# In the notation of [Graphs: Notation for This Lecture](lecture11_algorithms.ipynb#graph-notation),
# the problem is: given a directed graph with distances $c_{ij} \ge 0$ on the arcs,
# find a shortest path from a source node $s$ to a destination node $d$. An
# undirected edge, which can be used in both directions, counts as two arcs.
#
# As an example, we look for a shortest path from $s = \text{A}$ to
# $d = \text{F}$ in the graph of [](#fig-sp-example). The distance from D to F is
# $c_{DF} = 5$, and the edge between B and C can be used in both directions:
# $c_{BC} = c_{CB} = 2$.

# %% label="sp-example" tags=["remove-cell"]
positions = {
    "A": (0, 1),
    "B": (1.2, 2),
    "C": (1.2, 0),
    "D": (2.8, 2),
    "E": (2.8, 0),
    "F": (4, 1),
}
graph_arcs = {
    ("A", "B"): 2,
    ("A", "C"): 1,
    ("B", "C"): 2,
    ("B", "D"): 3,
    ("C", "D"): 3,
    ("C", "E"): 1,
    ("E", "D"): 1,
    ("D", "F"): 5,
    ("E", "F"): 2,
}
draw_graph(positions, graph_arcs, both_ways={("B", "C")})

# %% [markdown]
# :::{figure} #sp-example
# :label: fig-sp-example
#
# Example network with distances along the arcs. The edge between B and C can be
# used in both directions.
# :::

# %% [markdown]
# ## LO Approach
#
# ### Model
#
# The shortest path problem is a special case of the
# [transshipment problem](lecture9_transportation.ipynb#transshipment-problem): ship
# one unit from the source $s$ (supply 1) to the destination $d$ (demand 1) as
# cheaply as possible, where the cost of using an arc is its distance. The unit
# follows a path, and the cost is the length of that path.
#
# #### Decision Variables
#
# $x_{ij} = 1$ if the arc $i \to j$ is on the path, and 0 otherwise.
#
# #### Objective
#
# Minimize the length of the path, $\sum_{i,j} c_{ij} x_{ij}$, with
# $c_{ij} = \infty$ if there is no arc $i \to j$.
#
# #### Constraints
#
# One unit more leaves the source than enters it, one unit more enters the
# destination than leaves it, and every other node passes on what it receives (flow
# conservation). The constraint $x_{ij} \ge 0$ is needed: without it, a negative
# amount could flow backwards over an arc.
#
# We do not require $x_{ij} \in \{0, 1\}$. As for the transportation problem, the
# corner points of the feasible region are already integer, so the simplex method
# returns a 0/1 solution. If a solver does return a fractional solution, the unit is
# split over several paths, and each of these paths is then a shortest path.
#
# #### Complete LO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i,j} c_{ij} x_{ij} \\
# \text{s.t.} \quad & \sum_{j} x_{sj} - \sum_{j} x_{js} = 1 \\
# & \sum_{i} x_{id} - \sum_{j} x_{dj} = 1 \\
# & \sum_{i} x_{ik} = \sum_{j} x_{kj}, \quad \text{for all nodes } k \ne s, d \\
# & x_{ij} \ge 0, \quad \text{for all } i, j.
# \end{aligned}
# $$
#
# ### Model for the Example
#
# For [](#fig-sp-example), with source A and destination F, there is a variable for
# each of the 10 arcs; pairs of nodes without an arc have $c_{ij} = \infty$, so
# their variables are 0 in any optimal solution and can be left out. No arc enters A
# or leaves F, which simplifies their constraints:
#
# $$
# \begin{aligned}
# \min \quad & 2x_{AB} + x_{AC} + 2x_{BC} + 2x_{CB} + 3x_{BD} + 3x_{CD} + x_{CE}
#   + x_{ED} + 5x_{DF} + 2x_{EF} \\
# \text{s.t.} \quad & x_{AB} + x_{AC} = 1 & \text{(source A)} \\
# & x_{DF} + x_{EF} = 1 & \text{(destination F)} \\
# & x_{AB} + x_{CB} = x_{BC} + x_{BD} & \text{(node B)} \\
# & x_{AC} + x_{BC} = x_{CB} + x_{CD} + x_{CE} & \text{(node C)} \\
# & x_{BD} + x_{CD} + x_{ED} = x_{DF} & \text{(node D)} \\
# & x_{CE} = x_{ED} + x_{EF} & \text{(node E)} \\
# & x_{ij} \ge 0, \quad \text{for all arcs } i \to j.
# \end{aligned}
# $$
#
# ### Solving the Example in pulp
#
# The model is built as the transshipment model in
# [Transportation and Transshipment](lecture9_transportation.ipynb): the keys of the
# dictionary `distance` are the arcs, and a variable is only created for an arc that
# exists. The constraint of a node depends on whether it is the source, the
# destination, or another node.

# %%
distance = {
    ("A", "B"): 2,
    ("A", "C"): 1,
    ("B", "C"): 2,
    ("C", "B"): 2,
    ("B", "D"): 3,
    ("C", "D"): 3,
    ("C", "E"): 1,
    ("E", "D"): 1,
    ("D", "F"): 5,
    ("E", "F"): 2,
}
nodes = ["A", "B", "C", "D", "E", "F"]
source, destination = "A", "F"

shortest_path = pulp.LpProblem(name="shortest_path", sense=pulp.LpMinimize)
use = {
    (i, j): pulp.LpVariable(name=f"x_{i}_{j}", lowBound=0)
    for (i, j) in distance
}

shortest_path += pulp.lpSum(distance[a] * use[a] for a in distance)
for k in nodes:
    inflow = pulp.lpSum(use[i, j] for (i, j) in distance if j == k)
    outflow = pulp.lpSum(use[i, j] for (i, j) in distance if i == k)
    if k == source:
        shortest_path += outflow - inflow == 1, f"source_{k}"
    elif k == destination:
        shortest_path += inflow - outflow == 1, f"destination_{k}"
    else:
        shortest_path += inflow == outflow, f"conservation_{k}"

shortest_path.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[shortest_path.status])
print("arcs used:", [a for a in distance if use[a].value() > 0.5])
print("length:", shortest_path.objective.value())

# %% [markdown]
# The shortest path is A → C → E → F, with length $1 + 1 + 2 = 4$. The model has a
# variable for every arc and a constraint for every node. For a network with $n$
# nodes where (almost) every pair of nodes is connected, that is about $n^2$
# variables. Dijkstra's algorithm finds the answer much faster.

# %% [markdown]
# ## Dijkstra's Algorithm
#
# Edsger Dijkstra (1930-2002), a Dutch computer scientist who worked at the CWI in
# Amsterdam, published his algorithm in 1959. It finds the shortest paths from the
# source to all nodes at once, and it requires that all distances are non-negative,
# $c_{ij} \ge 0$.
#
# The idea is to grow a set $W$ of visited nodes, one node at a time. In every step
# we visit the nearest unvisited node, nearest in the sense of the shortest path
# that only passes through visited nodes. Dijkstra proved that once a node is
# visited, we know a shortest path from the source to that node.

# %% [markdown]
# (dijkstra-d-w)=
# ### What the Algorithm Keeps Track Of
#
# To make every intermediate step precise, we define for a set of visited nodes $W$
# and every node $x$:
#
# $$
# d_W(x) = \text{length of a shortest path from the source to } x
# \text{ that only uses intermediate nodes from } W.
# $$
#
# "Intermediate" means the nodes between the source and $x$; the source and $x$
# themselves need not be in $W$. If there is no such path, $d_W(x) = \infty$. For
# example, in [](#fig-sp-example) with $W = \{\text{A}\}$ only the direct arcs from A
# count: $d_W(\text{B}) = 2$, $d_W(\text{C}) = 1$ and $d_W(x) = \infty$ for the other
# nodes. With $W = \{\text{A}, \text{C}\}$, the path A → C → E is allowed too, so
# $d_W(\text{E}) = 2$.
#
# As $W$ grows, more paths are allowed and $d_W(x)$ can only decrease. When $W$
# contains all nodes, every path is allowed, so our aim is
# $d_{\text{all nodes}}(\text{destination})$.

# %% [markdown]
# (dijkstra-steps)=
# ### The Algorithm
#
# Write $W \setminus x'$ for the set $W$ without the node $x'$.
#
# - **Start:** $W = \emptyset$, $d_W(\text{source}) = 0$ and $d_W(x) = \infty$ for all
#   other nodes $x$.
# - **While** $W$ does not contain all nodes:
#   1. Find the node $x \notin W$ with the smallest $d_W(x)$; call it $x'$.
#   2. Visit $x'$: add it to $W$. This makes $x'$ the current node.
#   3. Update, for all $x \notin W$:
#
#      $$
#      d_W(x) = \min\{d_{W \setminus x'}(x') + c_{x'x},\ d_{W \setminus x'}(x)\}.
#      $$
#
#      When the first term is the smaller one, save $x'$ as the previous node of $x$.
#
# The update considers the two possibilities for a shortest path to $x$ now that
# $x'$ may be used as an intermediate node: the path goes via $x'$ (first reach $x'$,
# then take the arc $x' \to x$), or it does not use $x'$ and the old value stays. Only
# arcs out of $x'$ can give an improvement, so in practice we only check the
# unvisited neighbors of the current node, with $c_{x'x} = \infty$ for the others.
#
# The previous nodes give the shortest paths themselves: follow them backwards from
# the destination to the source.

# %% [markdown]
# ### Example
#
# We apply the algorithm to [](#fig-sp-example). [](#tbl-dijkstra) shows $d_W(x)$
# throughout the algorithm, with the previous node in brackets. Each row shows the
# values after updating, for the nodes that are not yet visited; the column
# "current" gives the node with the smallest value, which is visited next.
#
# :::{table} Distances $d_W(x)$ throughout Dijkstra's algorithm, from A, for the network in [](#fig-sp-example).
# :label: tbl-dijkstra
#
# | step | A | B | C | D | E | F | current | $W$ |
# |---|---|---|---|---|---|---|---|---|
# | start | 0 | ∞ | ∞ | ∞ | ∞ | ∞ | A | $\{\}$ |
# | 1 | | 2 (A) | 1 (A) | ∞ | ∞ | ∞ | C | $\{A\}$ |
# | 2 | | 2 (A) | | 4 (C) | 2 (C) | ∞ | B | $\{A, C\}$ |
# | 3 | | | | 4 (C) | 2 (C) | ∞ | E | $\{A, C, B\}$ |
# | 4 | | | | 3 (E) | | 4 (E) | D | $\{A, C, B, E\}$ |
# | 5 | | | | | | 4 (E) | F | $\{A, C, B, E, D\}$ |
# | 6 | | | | | | | | $\{A, C, B, E, D, F\}$ |
# :::
#
# Step by step:
#
# - **Start.** Only the source has a finite value, so A becomes current.
# - **Step 1.** Visit A, so $W = \{\text{A}\}$. Update the neighbors of A:
#   $d_W(\text{B}) = \min\{0 + 2, \infty\} = 2$ and
#   $d_W(\text{C}) = \min\{0 + 1, \infty\} = 1$, both with previous node A. The
#   smallest value among the unvisited nodes is 1, so C becomes current.
# - **Step 2.** Visit C. Via C, B could be reached in $1 + 2 = 3$, which is worse
#   than 2, so B keeps its value. D gets $1 + 3 = 4$ and E gets $1 + 1 = 2$, both via
#   C. B and E are tied at 2; we pick B (any choice works).
# - **Step 3.** Visit B. Via B, D could be reached in $2 + 3 = 5 > 4$, so nothing
#   changes, and E becomes current.
# - **Step 4.** Visit E. Via E, D is reached in $2 + 1 = 3 < 4$, so
#   $d_W(\text{D}) = 3$ with previous node E. F gets $2 + 2 = 4$.
# - **Step 5.** Visit D. Via D, F could be reached in $3 + 5 = 8 > 4$, so F keeps 4
#   and becomes current.
# - **Step 6.** Visit F. All nodes are in $W$, so we stop.
#
# The value of a node in the row where it becomes current is its shortest distance
# from A: $d(\text{F}) = 4$. For example, the shortest path from A to D has length 3.
# Backtracking through the previous nodes gives the paths: the previous node of F is
# E, that of E is C, and that of C is A, so the shortest path from A to F is
# A → C → E → F, the same as pulp found. Together, the previous nodes form a tree of
# shortest paths from A to all nodes, shown in [](#fig-sp-tree).

# %% label="sp-tree" tags=["remove-cell"]
draw_graph(
    positions,
    graph_arcs,
    both_ways={("B", "C")},
    highlight={("A", "B"), ("A", "C"), ("C", "E"), ("E", "D"), ("E", "F")},
    node_notes={
        "A": ("0", -0.35, 0.15),
        "B": ("2", -0.3, 0.25),
        "C": ("1", -0.3, -0.25),
        "D": ("3", 0.3, 0.25),
        "E": ("2", 0.3, -0.25),
        "F": ("4", 0.1, -0.35),
    },
)

# %% [markdown]
# :::{figure} #sp-tree
# :label: fig-sp-tree
#
# Result of Dijkstra's algorithm from A: next to each node its shortest distance from
# A, and in red the arcs from each node's previous node, which form the tree of
# shortest paths.
# :::

# %% [markdown]
# ### Why It Works
#
# Suppose $x'$ has the smallest value $d_W(x')$ among the unvisited nodes. Any path
# from the source to $x'$ that is not counted in $d_W(x')$ leaves the visited nodes at
# some point: it reaches a first node $y$ outside $W$, and then continues to $x'$.
# The part up to $y$ is at least $d_W(y) \ge d_W(x')$ long, and because all
# distances are non-negative, the rest of the path cannot make it shorter. So no
# path to $x'$ is shorter than $d_W(x')$, and visiting $x'$ fixes its shortest
# distance for good.
#
# This argument fails with negative distances: a path could first go to a far-away
# node and then come back over an arc with a large negative distance. That is why
# Dijkstra's algorithm requires $c_{ij} \ge 0$. The argument also shows that we may
# stop as soon as the destination is visited: we then already know its shortest
# distance.
#
# :::{exercise}
# :label: ex-sp-negative
#
# Give a small network with one negative distance for which Dijkstra's algorithm
# does not find the shortest path. Which step of the argument above fails?
# :::

# %% [markdown]
# ### Dijkstra's Algorithm in Python
#
# The function below follows the algorithm line by line. You do not have to be able
# to write it yourself, but read it next to [The Algorithm](#dijkstra-steps) and
# check that you can follow each step. The dictionary `d` holds $d_W(x)$ and the
# list `W` the visited nodes. To show the steps, the function also builds the table
# of [](#tbl-dijkstra).


# %%
def dijkstra(nodes, distance, source):
    """Shortest distances from source with Dijkstra's algorithm.

    Returns d (node -> shortest distance), previous (node -> previous node
    on a shortest path) and the table of all steps.
    """
    d = {x: math.inf for x in nodes}
    d[source] = 0
    previous = {}
    W = []
    rows = []
    while len(W) < len(nodes):
        # 1. find the unvisited node with the smallest d_W(x)
        current = min((x for x in nodes if x not in W), key=lambda x: d[x])
        rows.append(table_row(nodes, d, previous, W, current))
        # 2. visit it
        W.append(current)
        # 3. update the unvisited nodes via the current node
        for x in nodes:
            if x not in W and (current, x) in distance:
                via_current = d[current] + distance[current, x]
                if via_current < d[x]:
                    d[x] = via_current
                    previous[x] = current
    rows.append(table_row(nodes, d, previous, W, ""))
    table = pd.DataFrame(rows)
    table.index = ["start"] + [f"step {k}" for k in range(1, len(rows))]
    return d, previous, table


def table_row(nodes, d, previous, W, current):
    """One row of the table: d_W(x) (previous node) for unvisited x."""
    row = {}
    for x in nodes:
        if x in W:
            row[x] = ""
        elif x in previous:
            row[x] = f"{d[x]} ({previous[x]})"
        else:
            row[x] = "∞" if d[x] == math.inf else str(d[x])
    row["current"] = current
    row["W"] = "{" + ", ".join(W) + "}"
    return row


d, previous, table = dijkstra(nodes, distance, source)
table

# %% [markdown]
# Backtracking through `previous` gives the shortest path:

# %%
path = [destination]
while path[0] != source:
    path.insert(0, previous[path[0]])
print("shortest path:", " -> ".join(path), "with length", d[destination])

# %% tags=["remove-cell"] label="sp-exercise"
draw_graph(
    {"A": (0, 1), "B": (1.2, 2), "C": (1.2, 0), "D": (3.2, 2), "E": (3.2, 0)},
    {
        ("A", "B"): 10,
        ("A", "C"): 2,
        ("A", "D"): 21,
        ("B", "D"): 5,
        ("C", "B"): 3,
        ("C", "E"): 11,
        ("D", "E"): 2,
    },
    # keep the labels of the crossing arcs A -> D and C -> B apart
    label_at={("A", "D"): 0.65, ("C", "B"): 0.3},
)

# %% [markdown]
# :::{exercise}
# :label: ex-sp-dijkstra
#
# Find the shortest path from A to E in [](#fig-shortest-path-exercise) using
# Dijkstra's algorithm. Write down the table with $d_W(x)$ and the previous nodes,
# as in [](#tbl-dijkstra). Formulate the problem also as an LO model, solve it with
# pulp, and check that the two answers agree. Finally, check your table with the
# `dijkstra` function above.
# :::
#
# :::{figure} #sp-exercise
# :label: fig-shortest-path-exercise
#
# Directed graph of [](#ex-sp-dijkstra), with distances along the arcs.
# :::

# %% [markdown]
# ### Running Time
#
# How much work does Dijkstra's algorithm take for a network with $n$ nodes? Every
# repetition of the while loop visits one new node, so there are $n$ repetitions.
# In each repetition, finding the unvisited node with the smallest value and
# updating the unvisited nodes each look at no more than $n$ nodes. In total that is
# about $n \cdot n = n^2$ operations. For a network with 1000 nodes, this is in the
# order of a million operations, a fraction of a second on a computer: very fast. In
# [Complexity and Heuristics](lecture11_complexity-heuristics.ipynb) we compare this
# with problems for which no such fast algorithm is known.

# %% [markdown]
# :::{note} Navigation Software
# Navigation apps use variants of Dijkstra's algorithm. When you ask for the shortest
# route from A(msterdam) to B(erlin), however, Dijkstra's algorithm also computes the
# shortest routes to every place that is closer to Amsterdam than Berlin, in all
# directions, which is a waste of time.
#
# ![Illustration of a shortest-route calculation from Amsterdam to Berlin](images/lecture11_box7.2-map.png)
#
# A well-known improvement is the A* algorithm. It adds an estimate $h(x)$ of the
# remaining distance from $x$ to the destination, for example the straight-line
# distance, and visits the unvisited node with the smallest $d_W(x) + h(x)$ instead
# of the smallest $d_W(x)$. Nodes in the direction of the destination are then
# visited first. If $h(x)$ is never larger than the real remaining distance, A* still
# finds a shortest route, and it is much faster than Dijkstra's algorithm.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 7,
#   "Combinatorial Optimization" (shortest path).
# - Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
#   *Numerische Mathematik*, 1, 269-271.
