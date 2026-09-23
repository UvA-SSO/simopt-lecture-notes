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
# # Lecture 11: Exercises
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture11_exercises.ipynb)

# %% [markdown]
# The smaller exercises embedded in
# [Algorithms and Heuristics](lecture11_algorithms-heuristics.ipynb) and
# [Complexity](lecture11_complexity.ipynb) check what you just read. This notebook
# collects the larger exercises for Lecture 11: independent problems worth more time.
#
# :::{warning} Try It Yourself First
# The homework exercises below are representative of what you can expect on the exam:
# solve them by hand, pen-and-paper, without pulp or a computer. Attempt each one
# yourself, or make a serious effort, before opening the answer. If you do not manage to
# solve it, look at the answer to help you continue. Once solved, come back at a later
# time and try it again without looking at the answer. As extra practice, you can also
# solve them with pulp.
# :::

# %% [markdown]
# ## Homework Exercises

# %% [markdown]
# :::{exercise}
# :label: hw-11-1
#
# We want the shortest route from node A to node E in the directed graph below (travel
# times in hours).
#
# :::{figure} images/lecture11_hw1-ex1-graph.png
# :label: fig-hw-11-1
#
# Directed graph for Exercise 1.
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

# %% [markdown]
#
# :::{solution} hw-11-2
# :label: sol-hw-11-2
# :class: dropdown
#
# a. See the figure below.
#
# :::{figure} images/lecture11_hw1-ex2-graph.png
# :label: fig-hw-11-2
#
# The directed graph with capacities.
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
#    \text{s.t.} \quad & x_{AB} \le 6,\ x_{AD} \le 2,\ x_{BC} \le 1,\ x_{CD} \le 2,\ x_{BE} \le 4,\ x_{DE} \le 4 \\
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
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-11-3
#
# We want to go from node A to node E in the directed graph below; edge labels are travel
# costs (e.g. $A \to B$ costs 2; $D \to E$ costs nothing).
#
# :::{figure} images/lecture11_hw1-ex3-graph.png
# :label: fig-hw-11-3
#
# Directed graph for Exercise 3.
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

# %% [markdown]
# :::{exercise}
# :label: hw-11-4
#
# Consider the maximum-flow problem below.
#
# :::{figure} images/lecture11_hw1-ex4-graph.png
# :label: fig-hw-11-4
#
# Directed graph for Exercise 4.
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

# %% [markdown]
#
# :::{solution} hw-11-5
# :label: sol-hw-11-5
# :class: dropdown
#
# a. See the figure below.
#
# :::{figure} images/lecture11_hw1-ex5-graph.png
# :label: fig-hw-11-5
#
# The directed graph with capacities.
# :::
#
# b. Let $x_{ij}$ be the flow on arc $(i, j)$. Then
#
#    $$
#    \begin{aligned}
#    \max \quad & x_{AB} + x_{AC} \\
#    \text{s.t.} \quad & x_{AB} \le 20,\ x_{AC} \le 10,\ x_{BC} \le 5,\ x_{BD} \le 20,\ x_{CE} \le 20,\ x_{DF} \le 10,\ x_{ED} \le 10,\ x_{EF} \le 25 \\
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
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §7.1, §7.3, §7.4 (shortest
#   path, maximum flow).
