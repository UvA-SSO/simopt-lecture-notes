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
# description: "Homework exercises for Lecture 11 on algorithms, heuristics and complexity."
# thumbnail: null
# ---
# # Lecture 11: Exercises
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture11_exercises.ipynb)

# %% [markdown]
# The smaller exercises embedded in the other Lecture 11 notebooks check what you
# just read. This notebook collects the larger exercises for Lecture 11: independent problems worth more time.
#
# :::{warning} Try It Yourself First
# The homework exercises below are representative of what you can expect on the exam:
# solve them by hand, pen-and-paper, without pulp or a computer. Attempt each one
# yourself, or make a serious effort, before opening the answer. If you do not manage to
# solve it, look at the answer to help you continue. Once solved, come back at a later
# time and try it again without looking at the answer. As extra practice, you can also
# solve them with pulp.
# :::

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


# %% [markdown]
# ## Homework Exercises

# %% tags=["remove-cell"] label="hw11-ex1"
draw_graph(
    {"A": (0, 1), "B": (1.2, 2), "C": (1.2, 0), "D": (2.8, 2), "E": (2.8, 0)},
    {
        ("A", "B"): 3,
        ("A", "C"): 1,
        ("B", "D"): 2,
        ("B", "E"): 4,
        ("C", "B"): 1,
        ("C", "E"): 5,
        ("D", "E"): 1,
    },
)

# %% [markdown]
# :::{exercise}
# :label: hw-11-1
#
# We want the shortest route from node A to node E in the directed graph below (travel
# times in hours).
#
# :::{figure} #hw11-ex1
# :label: fig-hw-11-1
#
# Directed graph of [](#hw-11-1), with travel times in hours along the arcs.
# :::
#
# a. Solve this using Dijkstra's algorithm.
#
# b. Give the linear optimization formulation of this problem.
#
# c. Change the LO formulation from part b to find the *longest* path from A to E that
#    does *not* visit node C.
# :::
#

# %% [markdown]
#
# :::{solution} hw-11-1
# :label: sol-hw-11-1
# :class: dropdown
#
# a. Running Dijkstra's algorithm from A (each row is one step; $d_W(x)$ is the shortest
#    distance found so far to $x$ using only intermediate nodes in $W$, with the node
#    chosen as current underlined):
#
#    | step | A | B | C | D | E | current | $W$ |
#    |---|---|---|---|---|---|---|---|
#    | start | 0 | $\infty$ | $\infty$ | $\infty$ | $\infty$ | A | $\{\}$ |
#    | 1 | | 3 (A) | 1 (A) | $\infty$ | $\infty$ | C | $\{A\}$ |
#    | 2 | | 2 (C) | | $\infty$ | 6 (C) | B | $\{A, C\}$ |
#    | 3 | | | | 4 (B) | 6 (C) | D | $\{A, C, B\}$ |
#    | 4 | | | | | 5 (D) | E | $\{A, C, B, D\}$ |
#
#    The shortest path is $A \to C \to B \to D \to E$, distance 5.
#
# b. Let $x_{ij} = 1$ if edge $(i, j)$ is traversed, 0 otherwise. The LO is
#
#    $$
#    \begin{aligned}
#    \min \quad & 3x_{AB} + x_{AC} + 2x_{BD} + 4x_{BE} + x_{CB} + 5x_{CE} + x_{DE} \\
#    \text{s.t.} \quad & x_{AB} + x_{AC} = 1 \ \text{(source)} \\
#    & x_{BE} + x_{CE} + x_{DE} = 1 \ \text{(destination)} \\
#    & x_{AB} + x_{CB} = x_{BD} + x_{BE} \ \text{(node B)} \\
#    & x_{AC} = x_{CB} + x_{CE} \ \text{(node C)} \\
#    & x_{BD} = x_{DE} \ \text{(node D)} \\
#    & x_{ij} \ge 0 \text{ for all } i, j.
#    \end{aligned}
#    $$
#
#    Because the shortest-path problem has this special structure, requiring $x_{ij}$ to
#    be binary is not necessary; the destination constraint is automatically satisfied by
#    the others.
#
# c. Turn the objective into a maximization and add $x_{AC} = 0$ (forbidding node C).
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-11-2
#
# Consider a maximum-flow problem with source A, destination E, and capacities
# $A \to B: 6$, $A \to D: 2$, $B \to C: 1$, $C \to D: 2$, $B \to E: 4$, $D \to E: 4$.
#
# a. Draw the directed graph with capacities.
#
# b. Solve it using the Ford-Fulkerson algorithm and verify optimality.
#
# c. Give the LO formulation.
#
# We now have the option to augment the capacity of *one* arc by 2, at no extra cost.
#
# d. Extend the formulation of part c to allow for this capacity augmentation.
#
# e. Give the optimal solution of part d and a convincing argument why it is optimal.
# :::
#

# %% tags=["remove-cell"] label="hw11-ex2"
draw_graph(
    {"A": (0, 1), "B": (1.6, 2), "C": (1.6, 1), "D": (1.6, 0), "E": (3.2, 1)},
    {
        ("A", "B"): 6,
        ("A", "D"): 2,
        ("B", "C"): 1,
        ("B", "E"): 4,
        ("C", "D"): 2,
        ("D", "E"): 4,
    },
)

# %% [markdown]
#
# ::::{solution} hw-11-2
# :label: sol-hw-11-2
# :class: dropdown
#
# a. See the figure below.
#
# :::{figure} #hw11-ex2
# :label: fig-hw-11-2
#
# Solution of [](#hw-11-2), part a: the directed graph with capacities.
# :::
#
# b. Three augmenting paths suffice (different orders also work): $A \to B \to E$ with
#    flow 4; $A \to B \to C \to D \to E$ with flow 1; $A \to D \to E$ with flow 2. This
#    gives a total flow of 7. Cutting edges $(A, D)$, $(B, C)$, $(B, E)$ separates A from
#    E with total capacity $2 + 1 + 4 = 7$, equal to the flow found, so the flow of 7 is
#    optimal.
#
# c. Let $x_{ij}$ be the flow on arc $(i, j)$. Then
#
#    $$
#    \begin{aligned}
#    \max \quad & x_{AB} + x_{AD} \\
#    \text{s.t.} \quad & x_{AB} \le 6 \\
#    & x_{AD} \le 2 \\
#    & x_{BC} \le 1 \\
#    & x_{CD} \le 2 \\
#    & x_{BE} \le 4 \\
#    & x_{DE} \le 4 \\
#    & x_{AB} = x_{BC} + x_{BE} \ \text{(node B)} \\
#    & x_{BC} = x_{CD} \ \text{(node C)} \\
#    & x_{AD} + x_{CD} = x_{DE} \ \text{(node D)} \\
#    & x_{ij} \ge 0 \text{ for all } i, j.
#    \end{aligned}
#    $$
#
# d. Let binary $y_{ij} = 1$ if arc $(i, j)$ receives the extra capacity of 2. Replace
#    each capacity constraint $x_{ij} \le c_{ij}$ with $x_{ij} \le c_{ij} + 2y_{ij}$, and
#    add $\sum_{i,j} y_{ij} = 1$ so only one arc is augmented.
#
# e. The extra capacity is best used on one of the min-cut edges $(A, D)$, $(B, C)$, or
#    $(B, E)$: augmenting any one of these raises the flow to 8 (augmenting any other arc
#    leaves the same bottleneck cut unchanged, so the flow stays at 7).
# ::::

# %% tags=["remove-cell"] label="hw11-ex3"
draw_graph(
    {"A": (0, 1), "B": (1.2, 2), "C": (1.2, 0), "D": (2.6, 1), "E": (3.8, 1)},
    {
        ("A", "B"): 2,
        ("A", "C"): 4,
        ("B", "C"): 1,
        ("B", "D"): 6,
        ("B", "E"): 5,
        ("C", "D"): 3,
        ("C", "E"): 2,
        ("D", "E"): 0,
    },
    # move the labels of the arcs into D and E apart
    label_at={
        ("B", "D"): 0.4,
        ("C", "D"): 0.4,
        ("B", "E"): 0.7,
        ("C", "E"): 0.7,
    },
)

# %% [markdown]
# :::{exercise}
# :label: hw-11-3
#
# We want to go from node A to node E in the directed graph below; edge labels are travel
# costs (e.g. $A \to B$ costs 2; $D \to E$ costs nothing).
#
# :::{figure} #hw11-ex3
# :label: fig-hw-11-3
#
# Directed graph of [](#hw-11-3), with travel costs along the arcs.
# :::
#
# a. Find the cheapest path from A to E using Dijkstra's algorithm. Show your steps.
#
# b. Give an LO problem for finding the cheapest path from A to E.
#
# c. Show that the cheapest path from part a is a feasible solution of the LO from part b.
#
# d. Give an LO problem for finding the *longest* path that visits node D.
# :::
#

# %% [markdown]
#
# :::{solution} hw-11-3
# :label: sol-hw-11-3
# :class: dropdown
#
# a. Dijkstra's algorithm from A (previous node on the shortest path in square brackets):
#
#    | step | A | B | C | D | E | current | $W$ |
#    |---|---|---|---|---|---|---|---|
#    | start | 0 | $\infty$ | $\infty$ | $\infty$ | $\infty$ | A | $\{\}$ |
#    | 1 | | 2 [A] | 4 [A] | $\infty$ | $\infty$ | B | $\{A\}$ |
#    | 2 | | | 3 [B] | 8 [B] | 7 [B] | C | $\{A, B\}$ |
#    | 3 | | | | 6 [C] | 5 [C] | E | $\{A, B, C\}$ |
#
#    Reading backward from E gives the cheapest path $A \to B \to C \to E$, cost 5.
#
# b. Let $x_{ij} = 1$ if edge $(i, j)$ is traversed, 0 otherwise. The LO is
#
#    $$
#    \begin{aligned}
#    \min \quad & 2x_{AB} + 4x_{AC} + x_{BC} + 6x_{BD} + 5x_{BE} + 3x_{CD} + 2x_{CE} \\
#    \text{s.t.} \quad & x_{AB} + x_{AC} = 1 \ \text{(source A)} \\
#    & x_{AB} = x_{BC} + x_{BD} + x_{BE} \ \text{(node B)} \\
#    & x_{AC} + x_{BC} = x_{CD} + x_{CE} \ \text{(node C)} \\
#    & x_{BD} + x_{CD} = x_{DE} \ \text{(node D)} \\
#    & x_{BE} + x_{CE} + x_{DE} = 1 \ \text{(destination E)} \\
#    & x_{ij} \ge 0 \text{ for all } i, j.
#    \end{aligned}
#    $$
#
#    As in Exercise 1, requiring $x_{ij}$ to be binary is not necessary here.
#
# c. Path $A \to B \to C \to E$ corresponds to $x_{AB} = x_{BC} = x_{CE} = 1$ and 0
#    otherwise. Substituting into each constraint: $1 = 1$ (source), $1 = 1 + 0 + 0$
#    (node B), $0 + 1 = 0 + 1$ (node C), $0 + 0 = 0$ (node D), $0 + 1 + 0 = 1$
#    (destination) — all satisfied, and together with nonnegativity this shows the
#    solution is feasible.
#
# d. Replace the $\min$ in part b with $\max$, and add $x_{DE} = 1$ (equivalently,
#    $x_{BD} + x_{CD} = 1$) to force the path through D.
# :::

# %% tags=["remove-cell"] label="hw11-ex4"
draw_graph(
    {
        "A": (0, 1),
        "B": (1.2, 2),
        "C": (1.2, 0),
        "D": (2.8, 2),
        "E": (2.8, 0),
        "F": (4, 1),
    },
    {
        ("A", "B"): 3,
        ("A", "C"): 2,
        ("B", "D"): 3,
        ("C", "B"): 1,
        ("C", "D"): 2,
        ("C", "E"): 1,
        ("D", "E"): 2,
        ("D", "F"): 4,
        ("E", "F"): 2,
    },
)

# %% [markdown]
# :::{exercise}
# :label: hw-11-4
#
# Consider the maximum-flow problem below.
#
# :::{figure} #hw11-ex4
# :label: fig-hw-11-4
#
# Directed graph of [](#hw-11-4), with capacities along the arcs.
# :::
#
# a. Use the Ford-Fulkerson algorithm to find the optimal solution.
#
# b. Determine a minimal cut and give its value.
# :::
#

# %% [markdown]
#
# :::{solution} hw-11-4
# :label: sol-hw-11-4
# :class: dropdown
#
# a. Three augmenting paths suffice (different choices are also possible):
#    $A \to B \to D \to F$ with flow 3; $A \to C \to D \to F$ with flow 1;
#    $A \to C \to E \to F$ with flow 1. No further augmenting path exists, so the maximum
#    flow is 5.
#
# b. The edges $(A, B)$ and $(A, C)$ form a cut of capacity $3 + 2 = 5$, equal to the
#    maximum flow found, so it is a minimum cut.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-11-5
#
# Consider a directed graph with arc capacities $A \to B: 20$, $A \to C: 10$,
# $B \to C: 5$, $B \to D: 20$, $C \to E: 20$, $D \to F: 10$, $E \to D: 10$,
# $E \to F: 25$.
#
# a. Draw the directed graph with capacities.
#
# b. Formulate the LO problem that maximizes the total flow from A to F.
#
# c. Determine the maximum flow, and the flow on each arc.
#
# d. Give a convincing argument that this is indeed the maximum flow.
#
# e. Suppose at most 5 arcs may carry flow. Extend the LO formulation of part b to allow
#    for this.
# :::
#

# %% tags=["remove-cell"] label="hw11-ex5"
draw_graph(
    {
        "A": (0, 1),
        "B": (1.2, 2),
        "C": (1.2, 0),
        "D": (2.8, 2),
        "E": (2.8, 0),
        "F": (4, 1),
    },
    {
        ("A", "B"): 20,
        ("A", "C"): 10,
        ("B", "C"): 5,
        ("B", "D"): 20,
        ("C", "E"): 20,
        ("D", "F"): 10,
        ("E", "D"): 10,
        ("E", "F"): 25,
    },
)

# %% [markdown]
#
# ::::{solution} hw-11-5
# :label: sol-hw-11-5
# :class: dropdown
#
# a. See the figure below.
#
# :::{figure} #hw11-ex5
# :label: fig-hw-11-5
#
# Solution of [](#hw-11-5), part a: the directed graph with capacities.
# :::
#
# b. Let $x_{ij}$ be the flow on arc $(i, j)$. Then
#
#    $$
#    \begin{aligned}
#    \max \quad & x_{AB} + x_{AC} \\
#    \text{s.t.} \quad & x_{AB} \le 20 \\
#    & x_{AC} \le 10 \\
#    & x_{BC} \le 5 \\
#    & x_{BD} \le 20 \\
#    & x_{CE} \le 20 \\
#    & x_{DF} \le 10 \\
#    & x_{ED} \le 10 \\
#    & x_{EF} \le 25 \\
#    & x_{AB} = x_{BC} + x_{BD} \ \text{(node B)} \\
#    & x_{AC} + x_{BC} = x_{CE} \ \text{(node C)} \\
#    & x_{BD} + x_{ED} = x_{DF} \ \text{(node D)} \\
#    & x_{CE} = x_{ED} + x_{EF} \ \text{(node E)} \\
#    & x_{ij} \ge 0 \text{ for all } i, j.
#    \end{aligned}
#    $$
#
# c. Three augmenting paths suffice: $A \to B \to D \to F$ with flow 10;
#    $A \to B \to C \to E \to F$ with flow 5; $A \to C \to E \to F$ with flow 10. No
#    further augmenting path exists, giving a maximum flow of 25, with
#    $x_{AB} = 15$, $x_{AC} = 10$, $x_{BC} = 5$, $x_{BD} = 10$, $x_{CE} = 15$,
#    $x_{DF} = 10$, $x_{ED} = 0$, $x_{EF} = 15$.
#
# d. Edges $(A, C)$, $(B, C)$, and $(D, F)$ form a cut separating A from F, with capacity
#    $10 + 5 + 10 = 25$, equal to the flow found in part c, so that flow is optimal.
#
# e. Let binary $y_{ij} = 1$ if arc $(i, j)$ carries nonzero flow, and let $M$ be a big
#    constant. Add $x_{ij} \le M y_{ij}$ for every arc, and $\sum_{i,j} y_{ij} \le 5$.
# ::::

# %% tags=["remove-cell"] label="hw11-ex6"
draw_graph(
    {
        "A": (2.0, 2.35),
        "B": (3.52, 1.49),
        "C": (2.94, 0.09),
        "D": (1.06, 0.09),
        "E": (0.48, 1.49),
    },
    {
        ("A", "B"): 5,
        ("A", "C"): 8,
        ("A", "D"): 3,
        ("A", "E"): 2,
        ("B", "C"): 4,
        ("B", "D"): 6,
        ("B", "E"): 6,
        ("C", "D"): 7,
        ("C", "E"): 9,
        ("D", "E"): 5,
    },
    directed=False,
)

# %% [markdown]
# ::::{exercise}
# :label: hw-11-6
#
# The function `build_tour` below constructs a tour for the traveling salesman
# problem on the complete graph in [](#fig-hw-11-6).
#
# ```python
# distance = {
#     ("A", "B"): 5, ("A", "C"): 8, ("A", "D"): 3, ("A", "E"): 2,
#     ("B", "C"): 4, ("B", "D"): 6, ("B", "E"): 6,
#     ("C", "D"): 7, ("C", "E"): 9,
#     ("D", "E"): 5,
# }
#
#
# def edge_length(i, j):
#     return distance.get((i, j), distance.get((j, i)))
#
#
# def build_tour(nodes, start):
#     tour = [start]
#     unvisited = [node for node in nodes if node != start]
#     while unvisited:
#         current = tour[-1]
#         chosen = unvisited[0]
#         for node in unvisited:
#             if edge_length(current, node) < edge_length(current, chosen):
#                 chosen = node
#         print(current, "->", chosen, "length", edge_length(current, chosen))
#         tour.append(chosen)
#         unvisited.remove(chosen)
#     edges = zip(tour, tour[1:] + tour[:1])
#     length = sum(edge_length(i, j) for i, j in edges)
#     print("tour", " -> ".join(tour + tour[:1]), "length", length)
#     return tour, length
#
#
# build_tour(["A", "B", "C", "D", "E"], "A")
# ```
#
# :::{figure} #hw11-ex6
# :label: fig-hw-11-6
#
# Complete graph of [](#hw-11-6), with distances along the edges.
# :::
#
# a. Without running the code, determine what it prints.
#
# b. Describe in your own words what `build_tour` does. Does it always return a
#    shortest tour? Motivate your answer.
#
# c. How does the running time of `build_tour` grow with the number of nodes $n$?
#    Is it a polynomial algorithm?
#
# d. Propose two ways to improve the tours that `build_tour` finds, and say how each
#    changes the running time. Apply one of them to [](#fig-hw-11-6).
# ::::

# %% [markdown]
#
# :::{solution} hw-11-6
# :label: sol-hw-11-6
# :class: dropdown
#
# a. The code prints
#
#    ```text
#    A -> E length 2
#    E -> D length 5
#    D -> B length 6
#    B -> C length 4
#    tour A -> E -> D -> B -> C -> A length 25
#    ```
#
#    The last edge, C-A of length 8, is not printed in the loop but is part of the
#    length $2 + 5 + 6 + 4 + 8 = 25$.
#
# b. Starting at `start`, the function repeatedly travels to the closest node that
#    has not been visited yet, until all nodes are visited, and then returns to the
#    start. This is the **nearest neighbor heuristic**. It does not always return a
#    shortest tour: B → C → D → A → E → B has length $4 + 7 + 3 + 2 + 6 = 22 < 25$.
#    Every step takes the shortest edge available at that moment, without looking at
#    what that choice means for later steps. In the example, the cheap edges A-E and
#    E-D leave the expensive edge C-A of length 8 for the end. So it is a heuristic,
#    not an exact algorithm.
#
# c. The while loop runs $n - 1$ times, once for every node added to the tour. In
#    each repetition the for loop looks at all unvisited nodes, at most $n - 1$, and
#    removing the chosen node also takes at most $n$ steps. In total that is in the
#    order of $n \cdot n = n^2$ operations, $O(n^2)$, so the algorithm is
#    polynomial. Compare this with the $n!$ growth of trying all tours.
#
# d. Some possible improvements:
#
#    - Run `build_tour` from every start node, or from several random start nodes,
#      and keep the shortest tour. From every start node this takes $n$ times as
#      long, $O(n^3)$. For [](#fig-hw-11-6), the start nodes A to E give lengths
#      25, 22, 23, 22 and 24, so starting in B (or D) gives the tour
#      B → C → D → A → E → B of length 22, which is optimal here.
#    - Look two nodes ahead: choose the next node $j$ such that
#      $d_{\text{current},j} + d_{jk}$ is smallest over all pairs of unvisited nodes
#      $j \ne k$. Each step
#      then looks at in the order of $n^2$ pairs, so the running time becomes
#      $O(n^3)$. From A, the pair E, D (length $2 + 5 = 7$) gives E first; from E,
#      the pair B, C ($6 + 4 = 10$) gives B; then C and D follow. The tour
#      A → E → B → C → D → A has length $2 + 6 + 4 + 7 + 3 = 22$.
#    - Use the tour as initial tour for the
#      [2-opt heuristic](lecture11_complexity-heuristics.ipynb#local-search-heuristic).
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §7.1, §7.3, §7.4 (shortest
#   path, maximum flow).
