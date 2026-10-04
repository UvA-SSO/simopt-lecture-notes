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
# description: "The traveling salesman problem: an ILO model with subtour elimination constraints, solved in pulp, and the exact algorithm that tries all tours."
# thumbnail: null
# ---
# # Lecture 11: Traveling Salesman Problem
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture11_tsp.ipynb)

# %% [markdown]
# The third classical problem of this lecture is the traveling salesman problem
# (TSP): find the shortest tour along a number of cities. As for the
# [shortest path](lecture11_shortest-path.ipynb) and
# [maximum flow](lecture11_maximum-flow.ipynb) problems, we define the problem and
# give an example, solve the example with an ILO approach, and then look for a
# dedicated algorithm. This time the story ends differently. The ILO model needs a
# huge number of constraints, and the only exact algorithm we find tries all tours,
# which takes far too long for large instances. No relatively fast exact algorithm
# is known for the TSP. The next notebook,
# [Complexity and Heuristics](lecture11_complexity-heuristics.ipynb), explains why,
# and continues with this example using a heuristic.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate the TSP as an ILO model and explain why subtour elimination
#   constraints are needed;
# - solve a small TSP in pulp by adding subtour elimination constraints only when
#   they are needed;
# - explain why trying all tours is not practical for large instances.

# %% [markdown]
# ## The Traveling Salesman Problem
#
# What is the shortest tour that visits each city once and returns to the city it
# started from? This question, the traveling salesman problem, comes up whenever a
# vehicle or a person makes a round trip: a delivery van that delivers orders and
# returns to the warehouse, an order picker who collects the items of an order in a
# warehouse, or a machine that drills holes in a circuit board. It is also part of
# many larger planning problems, such as vehicle routing with several vans.
#
# Given are $n$ nodes (cities) and a distance $d_{ij}$ between every pair of nodes
# $i$ and $j$. The goal is to find the shortest tour that visits all nodes and
# returns to the start node. Where to start does not matter: a tour is a cycle, so
# it returns to its start anyway. If two cities are not directly connected, we set
# their distance very large. If the shortest route between two cities passes through
# a third city, use the length of that route as $d_{ij}$: the shortest distances
# between all pairs follow from [Dijkstra's algorithm](lecture11_shortest-path.ipynb).
#
# As an example, take the network of [](#fig-tsp-example), where all edges can be
# used in both directions. The tour A → B → D → F → E → C → A has length
# $2 + 3 + 5 + 2 + 6 + 1 = 19$. Is there a shorter one?

# %% tags=["hide-input"]
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

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


# %% tags=["remove-cell"] label="tsp-example"
positions = {
    "A": (0, 1),
    "B": (1.2, 2),
    "C": (1.2, 0),
    "D": (2.8, 2),
    "E": (2.8, 0),
    "F": (4, 1),
}
example_distance = {
    ("A", "B"): 2,
    ("A", "C"): 1,
    ("B", "C"): 2,
    ("B", "D"): 3,
    ("B", "E"): 2,
    ("C", "D"): 3,
    ("C", "E"): 6,
    ("D", "E"): 1,
    ("D", "F"): 5,
    ("E", "F"): 2,
}
# the edges B-E and C-D cross in the middle: move their labels apart
crossing_labels = {("B", "E"): 0.72, ("C", "D"): 0.28}


def tour_edges(tour):
    """The edges of a tour given as a list of nodes."""
    return {(tour[k], tour[(k + 1) % len(tour)]) for k in range(len(tour))}


draw_graph(
    positions,
    example_distance,
    highlight=tour_edges(["A", "B", "D", "F", "E", "C"]),
    directed=False,
    label_at=crossing_labels,
)

# %% [markdown]
# :::{figure} #tsp-example
# :label: fig-tsp-example
#
# Example network with distances along the edges, and in red the tour
# A → B → D → F → E → C → A of length 19.
# :::

# %% [markdown]
# ## ILO Approach
#
# ### Model
#
# #### Decision Variables
#
# $x_{ij} = 1$ if the tour goes from $i$ directly to $j$, and 0 otherwise, for
# $i, j = 1, \dots, n$ with $i \ne j$.
#
# #### Objective
#
# Minimize the length of the tour, $\sum_{i,j} d_{ij} x_{ij}$.
#
# #### Constraints
#
# The tour arrives in every city exactly once and leaves every city exactly once:
#
# $$
# \begin{aligned}
# & \sum_{i} x_{ij} = 1, \quad \text{for all } j \\
# & \sum_{i} x_{ji} = 1, \quad \text{for all } j.
# \end{aligned}
# $$
#
# These constraints are not enough. In [](#fig-tsp-subtours), every node is entered
# once and left once, but the red edges form two separate tours, A → B → C → A and
# D → E → F → D, instead of one tour along all nodes. Such separate tours are called
# **subtours**. Their total length is $5 + 8 = 13$, so a model with only the
# constraints above would happily return them.

# %% tags=["remove-cell"] label="tsp-subtours"
draw_graph(
    positions,
    example_distance,
    highlight=tour_edges(["A", "B", "C"]) | tour_edges(["D", "E", "F"]),
    directed=False,
    label_at=crossing_labels,
)

# %% [markdown]
# :::{figure} #tsp-subtours
# :label: fig-tsp-subtours
#
# Two subtours (red), A → B → C → A and D → E → F → D. Every node is entered and left
# exactly once, but this is not one tour.
# :::

# %% [markdown]
# To prevent subtours, look at a set $S$ of nodes and count how many times the tour
# goes directly from a node in $S$ to another node in $S$. A subtour along the
# nodes of $S$ alone does this $|S|$ times, where $|S|$ is the number of nodes in
# $S$: a tour along $z$ nodes has $z$ edges. One tour along all nodes, on the other
# hand, has to leave $S$ at some point, and then uses at most $|S| - 1$ edges
# inside $S$. So we add the **subtour elimination constraints**
#
# $$
# \sum_{i,j \in S} x_{ij} \le |S| - 1, \quad
# \text{for all node sets } S \text{ with } 2 \le |S| \le n - 1.
# $$
#
# In [](#fig-tsp-subtours), the set $S = \{\text{A}, \text{B}, \text{C}\}$ has
# $x_{AB} + x_{BC} + x_{CA} = 3 > |S| - 1 = 2$, so these subtours are no longer
# feasible. (Some formulations also allow $i = j$; the constraints with $|S| = 1$
# then forbid $x_{ii} = 1$.)
#
# #### Complete ILO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i,j} d_{ij} x_{ij} \\
# \text{s.t.} \quad & \sum_{i} x_{ij} = 1, \quad \text{for all } j \\
# & \sum_{i} x_{ji} = 1, \quad \text{for all } j \\
# & \sum_{i,j \in S} x_{ij} \le |S| - 1, \quad \text{for all } S \text{ with }
#   2 \le |S| \le n - 1 \\
# & x_{ij} \in \{0, 1\}, \quad \text{for all } i \ne j.
# \end{aligned}
# $$
#
# The price is the number of constraints: a set of $n$ nodes has $2^n$ subsets, so
# there are about $2^n$ subtour elimination constraints. For 50 cities that is
# $2^{50} \approx 10^{15}$ constraints, far too many to write down.
#
# ### Model for the Example
#
# The 10 edges of [](#fig-tsp-example) can each be used in both directions, which
# gives 20 variables; pairs of nodes without an edge get no variable. With the two
# directions of each edge grouped, the model is
#
# $$
# \begin{aligned}
# \min \quad & 2(x_{AB} + x_{BA}) + (x_{AC} + x_{CA}) + 2(x_{BC} + x_{CB})
#   + 3(x_{BD} + x_{DB}) + 2(x_{BE} + x_{EB}) \\
# & + 3(x_{CD} + x_{DC}) + 6(x_{CE} + x_{EC}) + (x_{DE} + x_{ED})
#   + 5(x_{DF} + x_{FD}) + 2(x_{EF} + x_{FE}) \\
# \text{s.t.} \quad & x_{BA} + x_{CA} = 1 & \text{(arrive in A)} \\
# & x_{AB} + x_{AC} = 1 & \text{(leave A)} \\
# & \quad \vdots & \text{(the same for B to F)} \\
# & x_{AB} + x_{BA} + x_{AC} + x_{CA} + x_{BC} + x_{CB} \le 2
#   & (S = \{\text{A}, \text{B}, \text{C}\}) \\
# & \quad \vdots & \text{(one for each } S \text{ with } 2 \le |S| \le 5) \\
# & x_{ij} \in \{0, 1\}, \quad \text{for all 20 variables.}
# \end{aligned}
# $$
#
# With 6 nodes there are $2^6 - 6 - 2 = 56$ sets $S$ with $2 \le |S| \le 5$, so 56
# subtour elimination constraints.

# %% [markdown]
# ### Solving the Example in pulp
#
# In practice, the subtour elimination constraints are added only when they are
# needed, as so-called lazy constraints: solve the model without them, look for
# subtours in the solution, add the constraint for the node set of each subtour, and
# solve again, until the solution is one tour. Usually only a small fraction of all
# subtour elimination constraints is ever added.
#
# In the code, `distance` holds each edge once; `arc_length` adds the reverse
# direction, so there is a variable for each of the 20 arcs. The function
# `find_subtours` follows the tour from node to node to split a solution into its
# subtours.

# %%
import pulp

distance = {
    ("A", "B"): 2,
    ("A", "C"): 1,
    ("B", "C"): 2,
    ("B", "D"): 3,
    ("B", "E"): 2,
    ("C", "D"): 3,
    ("C", "E"): 6,
    ("D", "E"): 1,
    ("D", "F"): 5,
    ("E", "F"): 2,
}
nodes = ["A", "B", "C", "D", "E", "F"]
# every edge can be used in both directions
reverse = {(j, i): dist for (i, j), dist in distance.items()}
arc_length = distance | reverse

tsp = pulp.LpProblem(name="tsp", sense=pulp.LpMinimize)
travel = {
    (i, j): pulp.LpVariable(name=f"x_{i}_{j}", cat="Binary")
    for (i, j) in arc_length
}
tsp += pulp.lpSum(arc_length[a] * travel[a] for a in arc_length)
for k in nodes:
    arrive = pulp.lpSum(travel[i, j] for (i, j) in arc_length if j == k)
    leave = pulp.lpSum(travel[i, j] for (i, j) in arc_length if i == k)
    tsp += arrive == 1, f"arrive_{k}"
    tsp += leave == 1, f"leave_{k}"


def find_subtours(travel):
    """Split the solution into its (sub)tours, as lists of nodes."""
    successor = {i: j for (i, j), var in travel.items() if var.value() > 0.5}
    subtours = []
    unvisited = set(successor)
    while unvisited:
        subtour = [min(unvisited)]
        while successor[subtour[-1]] != subtour[0]:
            subtour.append(successor[subtour[-1]])
        subtours.append(subtour)
        unvisited -= set(subtour)
    return subtours


tsp.solve(pulp.PULP_CBC_CMD(msg=False))
subtours = find_subtours(travel)
print(f"length {tsp.objective.value()}: {subtours}")
while len(subtours) > 1:
    for subtour in subtours:
        inside = pulp.lpSum(
            travel[i, j] for (i, j) in arc_length if {i, j} <= set(subtour)
        )
        tsp += inside <= len(subtour) - 1
    tsp.solve(pulp.PULP_CBC_CMD(msg=False))
    subtours = find_subtours(travel)
    print(f"length {tsp.objective.value()}: {subtours}")

# %% [markdown]
# The first solution consists of three subtours of two nodes, such as A → C → A:
# going to a neighbor and straight back also enters and leaves every node once. The
# constraint for $S = \{\text{A}, \text{C}\}$ is $x_{AC} + x_{CA} \le 1$. The second
# solution consists of the two subtours of [](#fig-tsp-subtours), with length 13.
# After adding their constraints too, the solver finds one tour of length 15. Only 5
# of the subtour elimination constraints were needed. For instances with thousands
# of cities, specialized solvers use this approach together with many other tricks,
# but the solution time can still grow very quickly with the number of cities.

# %% [markdown]
# ## Algorithm: Trying All Tours
#
# Is there a dedicated algorithm for the TSP, as Dijkstra's algorithm is for the
# shortest path problem? A simple exact algorithm, called brute force, tries every
# tour:
#
# - **Start:** best length $= \infty$.
# - **For** every tour:
#   1. Calculate the total distance of the tour.
#   2. If it is shorter than the best length so far, it becomes the new best tour.
#
# Fix the start node; then the tour can visit the other $n - 1$ nodes in any order.
# That gives $(n - 1)! = (n - 1) \cdot (n - 2) \cdots 2 \cdot 1$ possible tours. For
# the example with $n = 6$, that is $5! = 120$ tours, which a computer checks in no
# time:


# %%
import itertools
import math


def edge_length(i, j):
    """Distance between i and j, infinite if there is no edge."""
    return distance.get((i, j), distance.get((j, i), math.inf))


def tour_length(tour):
    """Total distance of a tour (list of nodes) back to its start."""
    return sum(edge_length(i, j) for i, j in zip(tour, tour[1:] + tour[:1]))


start, *others = nodes
best_tour: list[str] = []
best_length = math.inf
n_tours = 0
for order in itertools.permutations(others):
    tour = [start, *order]
    n_tours += 1
    if tour_length(tour) < best_length:
        best_tour, best_length = tour, tour_length(tour)
print(f"checked {n_tours} tours")
print(
    "shortest tour:", " -> ".join(best_tour + [start]), "length", best_length
)

# %% [markdown]
# The running time of this algorithm grows like $n!$, and that grows extremely fast.
# For 20 cities there are $19! \approx 1.2 \cdot 10^{17}$ tours. Even at a billion
# tours per second, checking them all takes almost four years, and every extra city
# multiplies that time by the number of cities. This is not practical for
# reasonably sized instances.
#
# Can we do better? Unlike for the shortest path and maximum flow problems, no
# relatively fast exact algorithm is known for the TSP: not via the ILO model, and
# not via any dedicated algorithm. The next notebook,
# [Complexity and Heuristics](lecture11_complexity-heuristics.ipynb), explains why
# there probably is none, and then solves this example with a heuristic.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 7,
#   "Combinatorial Optimization" (traveling salesman problem).
