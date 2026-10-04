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
# description: "Why some optimization problems (shortest path, maximum flow, LO) can be solved quickly while others (TSP, ILO in general) apparently cannot, and how heuristics such as 2-opt find good solutions for them."
# thumbnail: null
# ---
# # Lecture 11: Complexity and Heuristics
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture11_complexity-heuristics.ipynb)

# %% [markdown]
# In the previous notebooks we solved the
# [shortest path](lecture11_shortest-path.ipynb) and
# [maximum flow](lecture11_maximum-flow.ipynb) problems with fast exact algorithms,
# while for the [traveling salesman problem](lecture11_tsp.ipynb) we only found an
# exact algorithm that tries all tours, which takes far too long for large
# instances. Complexity theory is about the running times of algorithms, and it
# explains this difference. It also tells us what to expect in practice: whether a
# solver will find the optimal solution of a large instance, or whether a heuristic
# is the better choice. In the second part of this notebook we continue with the TSP
# example and solve it with a heuristic, the 2-opt heuristic.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - explain the difference between polynomial and non-polynomial running-time
#   growth, and why it matters in practice;
# - classify the problems of this course as belonging to P or being NP-complete,
#   and explain the practical consequence of that classification;
# - explain why LO is efficiently solvable in practice although the simplex method
#   is not polynomial;
# - distinguish metaheuristics from problem-specific heuristics;
# - apply the 2-opt heuristic by hand, and explain the difference between a local
#   and a global optimum.

# %% [markdown]
# ## Running Time
#
# The running time of an algorithm depends on the size of the instance, for example
# the number of nodes $n$ of a network. We count the number of basic operations the
# algorithm performs as a function of $n$, and are only interested in how fast it
# grows. Some examples from this lecture:
#
# - [Dijkstra's algorithm](lecture11_shortest-path.ipynb) repeats its loop $n$ times
#   and looks at no more than $n$ nodes per repetition, so it takes about $n^2$
#   operations;
# - [trying all tours](lecture11_tsp.ipynb) of a TSP means checking
#   $(n - 1)!$ tours, which takes about $n!$ operations.
#
# A precise count, for example $c \cdot n(n - 1)/2$ operations with $c$ the number
# of operations to update one node, does not matter much: for large $n$, the term
# $n^2$ determines the growth. We write that the running time is $O(n^2)$, "of the
# order $n^2$". An algorithm is **polynomial** if its running time is $O(n^k)$ for
# some fixed power $k$, and **non-polynomial** otherwise, for example if it grows
# like $2^n$ or $n!$.
#
# The difference is huge, as [](#tbl-complexity-growth) shows.
#
# :::{table} Values of polynomial and non-polynomial functions.
# :label: tbl-complexity-growth
#
# | $n$ | $\log_2 n$ | $n \log_2 n$ | $n^2$ | $n^5$ | $2^n$ | $n!$ |
# |---|---|---|---|---|---|---|
# | 1 | 0 | 0 | 1 | 1 | 2 | 1 |
# | 5 | 2.32 | 11.6 | 25 | 3125 | 32 | 120 |
# | 10 | 3.32 | 33.2 | 100 | $10^5$ | 1024 | $3.6 \cdot 10^6$ |
# | 20 | 4.32 | 86.4 | 400 | $3.2 \cdot 10^6$ | $1.0 \cdot 10^6$ | $2.4 \cdot 10^{18}$ |
# | 50 | 5.64 | 282 | 2500 | $3.1 \cdot 10^8$ | $1.1 \cdot 10^{15}$ | $3.0 \cdot 10^{64}$ |
# | 100 | 6.64 | 664 | $10^4$ | $10^{10}$ | $1.3 \cdot 10^{30}$ | $9.3 \cdot 10^{157}$ |
# | 1000 | 9.97 | 9966 | $10^6$ | $10^{15}$ | $1.1 \cdot 10^{301}$ | $4.0 \cdot 10^{2567}$ |
# :::
#
# Suppose a computer performs a billion ($10^9$) operations per second. An algorithm
# with $n^5$ operations then solves an instance with $n = 100$ in 10 seconds. An
# algorithm with $2^n$ operations takes about $4 \cdot 10^{13}$ years for the same
# instance, thousands of times the age of the universe, and one with $n!$
# operations takes even longer: there are only about $10^{80}$ atoms in the
# universe. Faster computers do not help much. If computer power doubles every two
# years, as Moore's law roughly says, a $2^n$ algorithm can handle one extra node
# every two years. A quantum computer could at best take the square root of these
# numbers for some problems, which still leaves $\sqrt{100!} \approx 3 \cdot 10^{78}$.

# %% [markdown]
# (p-np-complete)=
# ## P and NP-Complete
#
# Researchers try to find polynomial algorithms for every problem. For some problems
# they succeeded, and proved that the algorithm always finds an optimal solution.
# For other problems, such as the TSP, hundreds of researchers have looked for a
# polynomial algorithm for decades without success. Based on the running time of the
# best known exact algorithm, there are, roughly speaking, two classes of problems.
#
# - **P** is the class of problems for which a polynomial exact algorithm is known.
#   Optimal solutions can be found even for large instances. Examples are the
#   transportation problem, the shortest path problem, the maximum flow problem,
#   the product-mix problem and LO in general.
# - **NP-complete** problems are problems for which no polynomial exact algorithm
#   is known. Large instances are solved with [heuristics](#heuristics), or with
#   solvers that may need a very long time to prove optimality. Examples are the
#   knapsack problem, machine scheduling, set covering, the TSP and ILO in general.
#
# The NP-complete problems are, in a precise sense, all equally hard: each of them
# can be translated into any other in polynomial time. So a polynomial algorithm for
# one NP-complete problem would give a polynomial algorithm for all of them.
#
# For the modeling in this course, this means that adding integer variables to an LO
# model can change a problem that solves in a second into one that a solver cannot
# finish.
#
# :::{note} Complexity of Linear Optimization
# The simplex method, invented by Dantzig for solving LO problems, works very well
# in practice, even for very large problems. However, it is possible to construct
# problems for which its running time is not polynomial in the size, so the simplex
# method is not a polynomial algorithm. In the 1980s, interior-point methods were
# developed that solve LO problems in polynomial time, which proves that LO is in P.
# Solvers offer both methods.
# :::

# %% [markdown]
# ## How to Win a Million Dollars
#
# Whether NP-complete problems can be solved in polynomial time is an open question,
# known as "P versus NP". It is one of the seven Millennium Prize Problems, for each
# of which the Clay Mathematics Institute offers US$1 million. To win, do one of the
# following:
#
# 1. find a polynomial exact algorithm for an NP-complete problem, or
# 2. prove that no polynomial exact algorithm exists for an NP-complete problem.
#
# Most researchers believe that the second statement is true, but nobody has proved
# it. Of the seven Millennium Prize Problems, only the Poincaré conjecture has been
# officially solved so far, by Grigori Perelman in 2003. In September 2026, OpenAI
# claimed that its AI system had solved another one, the Navier-Stokes problem
# about the equations that describe how fluids flow
# ([Nature, 2026](https://www.nature.com/articles/d41586-026-02842-5)). At the time
# of writing, mathematicians are still checking the proof.

# %% [markdown]
# :::{note} Consequences for Artificial General Intelligence
# Artificial general intelligence (AGI) is AI at the level of humans. Training an AI
# model means searching a huge space of possible models for a good one, an
# optimization problem. If P ≠ NP, as most researchers believe, there is no fast
# exact algorithm for the hardest of these problems, whatever the computing power.
# At the size of the problems that "human-level" AI would need to solve, finding the
# best model may then be computationally intractable, in the same sense as the TSP.
# This is one argument for skepticism about how soon AGI will be reached.
# :::

# %% [markdown]
# (heuristics)=
# ## Heuristics
#
# For NP-complete problems such as the TSP, no polynomial exact algorithm is known. A
# heuristic does not aim for the optimum but for a near-optimal solution in a
# reasonable amount of time. There are two types:
#
# - **Metaheuristics** are not specific to one problem. Examples are local search,
#   evolutionary algorithms, tabu search and simulated annealing. They still need
#   some tailoring to the problem at hand, for example a definition of which
#   solutions are "close" to each other.
# - **Problem-specific heuristics** use the structure of one problem. For common
#   problems there is a lot of literature on them. The 2-opt heuristic for the
#   TSP, below, is an example.
#
# Before using a heuristic, ask whether you need the optimal solution at all, and
# how much worse than optimal a solution may be. A heuristic usually gives no
# guarantee about how far its solution is from the optimum, so it can be hard to
# tell whether a solution is good enough. Comparing heuristics on the same
# instances, or with a lower bound such as the value of an LO relaxation, helps.

# %% [markdown]
# (local-search-heuristic)=
# ## The 2-Opt Heuristic
#
# ### Local Search
#
# The 2-opt heuristic is a **local search** method. Local search starts with some
# solution and repeatedly moves to a better solution close to the current one, until
# there is no better solution close by. "Close" is made precise by a
# **neighborhood**: for every solution $T$, a set $N(T)$ of solutions that can be
# reached from $T$ with one small change. Writing $L(T)$ for the length of tour $T$,
# local search is:
#
# - **Start:** an initial tour $T$.
# - **While** $N(T)$ contains a tour $T'$ with $L(T') < L(T)$:
#   - $T = T'$.
#
# The final tour has no better tour in its neighborhood: it is a **local optimum**.
# There is no guarantee that it is a **global optimum**, a best tour overall.
#
# The 2-opt heuristic uses the TSP-specific neighborhood $N(T)$ of all tours that
# can be reached from $T$ with one 2-opt swap:
#
# 1. remove two edges of the tour that do not share a node;
# 2. reconnect the two pieces into a tour, with the other two edges that do so.
#
# There is only one other way to reconnect the pieces into one tour: if the tour
# goes $\dots \to a \to b \to \dots \to c \to d \to \dots$ and we remove the edges
# $a$-$b$ and $c$-$d$, the new tour goes
# $\dots \to a \to c \to \dots \to b \to d \to \dots$, where the part from $b$ to $c$
# is now traveled in the opposite direction. The length changes by
#
# $$
# d_{ac} + d_{bd} - d_{ab} - d_{cd},
# $$
#
# so the swap is an improvement if this is negative. For the initial tour, a simple
# choice is to start at some node and repeatedly move to a random unvisited node.

# %% [markdown]
# ### Example
#
# We continue with the example of
# [Traveling Salesman Problem](lecture11_tsp.ipynb), with the same distances and the
# same function `tour_length` as there. A missing edge has distance $\infty$.

# %%
import math

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


def edge_length(i, j):
    """Distance between i and j, infinite if there is no edge."""
    return distance.get((i, j), distance.get((j, i), math.inf))


def tour_length(tour):
    """Total distance of a tour (list of nodes) back to its start."""
    return sum(edge_length(i, j) for i, j in zip(tour, tour[1:] + tour[:1]))


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


def tour_edges(tour):
    """The edges of a tour given as a list of nodes."""
    return {(tour[k], tour[(k + 1) % len(tour)]) for k in range(len(tour))}


# %% tags=["remove-cell"] label="tsp-start"
positions = {
    "A": (0, 1),
    "B": (1.2, 2),
    "C": (1.2, 0),
    "D": (2.8, 2),
    "E": (2.8, 0),
    "F": (4, 1),
}
# the edges B-E and C-D cross in the middle: move their labels apart
crossing_labels = {("B", "E"): 0.72, ("C", "D"): 0.28}
draw_graph(
    positions,
    distance,
    highlight=tour_edges(["A", "B", "D", "F", "E", "C"]),
    directed=False,
    label_at=crossing_labels,
)

# %% [markdown]
# :::{figure} #tsp-start
# :label: fig-tsp-start
#
# The TSP example network with the initial tour A → B → D → F → E → C → A of length
# 19 (red).
# :::

# %% [markdown]
# We start with the tour A → B → D → F → E → C → A of length 19 in
# [](#fig-tsp-start). Consider the swap that removes the edges B-D (length 3) and
# E-C (length 6). Reconnecting gives the edges B-E (length 2) and D-C (length 3),
# and the length changes by $2 + 3 - 3 - 6 = -4$. The new tour
# A → B → E → F → D → C → A in [](#fig-tsp-2opt) has length 15: the part
# D → F → E is now traveled as E → F → D. No 2-opt swap improves this tour, so it is
# a local optimum. Here it is also a global optimum: in
# [Traveling Salesman Problem](lecture11_tsp.ipynb), pulp and brute force found the
# same length.

# %% tags=["remove-cell"] label="tsp-2opt"
draw_graph(
    positions,
    distance,
    highlight=tour_edges(["A", "B", "E", "F", "D", "C"]),
    directed=False,
    label_at=crossing_labels,
)

# %% [markdown]
# :::{figure} #tsp-2opt
# :label: fig-tsp-2opt
#
# The tour A → B → E → F → D → C → A of length 15 (red), found with one 2-opt swap
# from the tour in [](#fig-tsp-start).
# :::

# %% [markdown]
# ### 2-Opt in Python
#
# The function `better_neighbor` goes through all 2-opt swaps of a tour and returns
# the first tour in its neighborhood that is shorter, or `None` if there is none. A
# swap that removes the edges after positions `i` and `j` reverses the part of the
# tour in between. The loop at the bottom is the local search algorithm above.


# %%
def better_neighbor(tour):
    """First tour in the 2-opt neighborhood of tour that is shorter."""
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # these two edges share the start node
            neighbor = (
                tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
            )
            if tour_length(neighbor) < tour_length(tour):
                return neighbor
    return None


tour = ["A", "B", "D", "F", "E", "C"]
print("initial tour:", "".join(tour), "length", tour_length(tour))
neighbor = better_neighbor(tour)
while neighbor is not None:
    tour = neighbor
    print("better tour: ", "".join(tour), "length", tour_length(tour))
    neighbor = better_neighbor(tour)
print("local optimum:", "".join(tour))

# %% [markdown]
# ### Local and Global Optima
#
# How good the local optimum is depends on the initial tour and on the
# neighborhood. Starting from another initial tour can end in another local optimum,
# so a common approach is to run the heuristic from several initial tours and keep
# the best result. A larger neighborhood gives better local optima, but takes longer
# to search. The 3-opt heuristic removes three edges and tries all ways of
# reconnecting the pieces, and is known to give better tours than 2-opt. In general,
# the $k$-opt neighborhood removes $k$ edges and contains in the order of $n^k$
# tours.

# %% tags=["remove-cell"] label="tsp-exercise"
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
        ("A", "B"): 2,
        ("A", "C"): 3,
        ("A", "D"): 4,
        ("B", "C"): 2,
        ("B", "D"): 3,
        ("B", "E"): 2,
        ("C", "D"): 3,
        ("C", "E"): 3,
        ("D", "E"): 1,
        ("D", "F"): 5,
        ("E", "F"): 4,
    },
    highlight=tour_edges(["A", "B", "D", "F", "E", "C"]),
    directed=False,
    # move labels away from where edges cross
    label_at={("A", "D"): 0.7, ("B", "E"): 0.72, ("C", "D"): 0.28},
)

# %% [markdown]
# :::{exercise}
# :label: ex-tsp-2opt
#
# Construct a local optimum using 2-opt for [](#fig-tsp-distances), starting with
# the tour A → B → D → F → E → C → A. Then check your answer with
# `better_neighbor`, after changing `distance` to the distances in the figure.
# :::
#
# :::{figure} #tsp-exercise
# :label: fig-tsp-distances
#
# Undirected graph of [](#ex-tsp-2opt), with distances along the edges.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 7,
#   "Combinatorial Optimization" (complexity, heuristics).
