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
# # Lecture 11: Complexity

# %% [markdown]
# In the previous notebook we studied 3 archetypical CO problems: shortest path, max flow, and TSP (see [Algorithms and Heuristics](lecture11_algorithms-heuristics.ipynb)). We saw a crucial difference between them: shortest path and max flow have run times polynomial in the size of the problem, but TSP grows as a factorial in $n$ (which is faster than exponential). This really makes a difference. The following table illustrates that.
#
# :::{note} ILO Solution of the TSP
# We can try to construct an LO formulation for the TSP, just like we did for the shortest path and the max flow problem. With $c_{ij}$ we denote the length of the edge between $i$ and $j$ and $x_{ij}$ is the binary variable which indicates whether the edge $ij$ is included in the tour. Then the obvious LO formulation is:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i,j,\, i \ne j} c_{ij} x_{ij} \\
# \text{subject to} \quad & \sum_{i \ne k} x_{ik} = \sum_{j \ne k} x_{kj} = 1 \text{ for all nodes } k; \\
# & x_{ij} \in \{0, 1\} \text{ for all } i, j,\ i \ne j.
# \end{aligned}
# $$
#
# However, this problem does not always result in a tour. For example, when applied to the graph below, we get two disconnected tours with 3 nodes: ABC and DEF. Therefore we need additional constraints, for example requiring that any subset has fewer used edges than there are nodes in the subset:
#
# ![An undirected graph with distances](images/lecture11_fig7.6.png)
#
# $$
# \sum_{i,j \in S} x_{ij} \le |S| - 1 \text{ for all } S \subset V \text{ with } 2 \le |S| \le n-2.
# $$
#
# By adding this constraint, we can solve the TSP to optimality. However, there are $2^n$ subsets of $V$, leading to an exponential number of constraints. This makes this solution approach practically infeasible.
# :::
#
# | $n$ | $n^2$ | $n^3$ | $2^n$ | $n!$ |
# |---|---|---|---|---|
# | 10 | 100 | 1000 | 1024 | 3.6 × 10⁶ |
# | 100 | 10⁴ | 10⁶ | 1.3 × 10³⁰ | 9.3 × 10¹⁵⁷ |
# | 1000 | 10⁶ | 10⁹ | 1.1 × 10³⁰¹ | 4.0 × 10²⁵⁶⁷ |
#
# If a solution takes 1 µs to evaluate, then an algorithm with $n^3$ steps takes 1 second to evaluate for size 100; an algorithm with $n!$ steps would take many times the age of the earth... Even if Moore's law, which roughly states that computer power doubles every two years, continues to hold, it will take thousands of years before hardware is fast enough to make running times acceptable.
#
# Researchers are therefore always interested in finding an algorithm with polynomial complexity. For some problems (such as shortest path and max flow) these algorithms are found and mathematically proven to terminate with the optimal solution. For a group of other problems (such as the TSP) hundreds of researchers spend decades of their lives looking in vain for polynomial algorithms... This suggests that there are, roughly speaking, two classes of problems. P is the class of problems for which polynomial-time algorithms are known. NP-complete is a class of problems for which you can easily decide whether a proposed solution is indeed a solution, but for which there is no known polynomial-time algorithm to find the optimal solution. Shortest path and max flow are in P; TSP is an example of an NP-complete problem. Other NP-complete problems are the knapsack problem, machine scheduling and set covering.
#
# :::{note} Complexity of Linear Optimization
# The simplex method, which is the original method invented by Dantzig for solving LO problems, has been proven to work very well, even for very big problems. However, it is possible to construct problems for which the run time is not polynomial in the size, thus the simplex method is not in P. In the 1980s, the so-called interior-point methods were developed that solve LO in polynomial time. These are widely used since then.
# :::
#
# It is important to note that there are more difficult classes of problems than NP-complete problems, for example those where it is hard to verify whether a proposed solution is indeed feasible. Note also that solving a mathematical optimization problem is only part of solving business problems. Indeed, translating a business problem into a mathematical model — modeling — is often harder than solving the resulting model. It is the ultimate goal of business analytics to solve any business problem in a rational data-driven way. Many of these problems, especially the strategic ones ("which product to develop?" or "how to maximize profit while keeping the risk of a loss below 5%?", or even "which employees to hire?") are very hard to model. Therefore modeling is an essential part of business analytics. Solving these types of problems in a rational mathematical way was the promise of management science in the 1950s. This failed and OR/MS was largely focused on operational problems in the following decades. The availability of data brings the solution to these problems within reach.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 7, "Combinatorial Optimization," §7.4.
