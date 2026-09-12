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
# In the previous notebook we studied 3 archetypical CO problems: shortest path, max flow, and TSP (see [Algorithms and Heuristics](lecture11_algorithms-heuristics.ipynb)), including an ILO formulation of the TSP with an exponential number of subtour elimination constraints. We saw a crucial difference between these problems: shortest path and max flow have run times polynomial in the size of the problem, but TSP grows as a factorial in $n$ (which is faster than exponential). This really makes a difference. [](#tbl-complexity-growth) illustrates that.
#
# :::{table} Growth rates of typical algorithm complexity classes.
# :label: tbl-complexity-growth
#
# | $n$ | $n^2$ | $n^3$ | $2^n$ | $n!$ |
# |---|---|---|---|---|
# | 10 | 100 | 1000 | 1024 | 3.6 × 10⁶ |
# | 100 | 10⁴ | 10⁶ | 1.3 × 10³⁰ | 9.3 × 10¹⁵⁷ |
# | 1000 | 10⁶ | 10⁹ | 1.1 × 10³⁰¹ | 4.0 × 10²⁵⁶⁷ |
# :::
#
# If a solution takes 1 µs to evaluate, then an algorithm with $n^3$ steps takes 1 second to evaluate for size 100; an algorithm with $n!$ steps would take many times the age of the earth... Even if Moore's law, which roughly states that computer power doubles every two years, continues to hold, it will take thousands of years before hardware is fast enough to make running times acceptable.
#
# Researchers are therefore always interested in finding an algorithm with polynomial complexity. For some problems (such as shortest path and max flow) these algorithms are found and mathematically proven to terminate with the optimal solution. For a group of other problems (such as the TSP) hundreds of researchers spend decades of their lives looking in vain for polynomial algorithms... This suggests that there are, roughly speaking, two classes of problems. **P** is the class of problems for which polynomial-time algorithms are known — besides shortest path and max flow, this includes the product-mix and LO problems from earlier notebooks: an exact solution can be found quickly even for large instances. **NP-complete** is a class of problems for which you can easily decide whether a proposed solution is indeed a solution, but for which there is no known polynomial-time algorithm to find the optimal solution, so large instances are tackled with heuristics instead. Besides the TSP, the knapsack problem, machine scheduling, set covering, and ILO problems in general are NP-complete.
#
# :::{note} Complexity of Linear Optimization
# The simplex method, which is the original method invented by Dantzig for solving LO problems, has been proven to work very well, even for very big problems. However, it is possible to construct problems for which the run time is not polynomial in the size, thus the simplex method is not in P. In the 1980s, the so-called interior-point methods were developed that solve LO in polynomial time. These are widely used since then.
# :::
#
# ## How to Win a Million Dollars
#
# The P versus NP-complete distinction is not merely a practical curiosity: it is one of the seven Millennium Prize Problems, for which the Clay Mathematics Institute has offered a US$1 million reward. To collect it, do either of the following and have your proof checked:
#
# 1. find a polynomial-time exact algorithm that solves an NP-complete problem, or
# 2. prove that no such polynomial-time algorithm can exist.
#
# Most researchers believe the second statement is the true one, but nobody has managed to prove it — the "P versus NP" question remains open. (Of the seven Millennium Problems, only the Poincaré conjecture has so far been solved, by Grigori Perelman in 2003.)
#
# :::{note} A Note on Artificial General Intelligence
# If P ≠ NP, as is widely believed, then no fast exact algorithm exists for solving an NP-complete problem, no matter how much computing power becomes available. To the extent that "human-level" artificial general intelligence (AGI) requires searching an astronomically large space of possible models at the scale needed to match human performance, this is a reason for skepticism that AGI is imminent: the underlying search problem may simply be computationally intractable, in the same sense the TSP is — no matter how fast hardware gets.
# :::
#
# It is important to note that there are more difficult classes of problems than NP-complete problems, for example those where it is hard to verify whether a proposed solution is indeed feasible. Note also that solving a mathematical optimization problem is only part of solving business problems. Indeed, translating a business problem into a mathematical model — modeling — is often harder than solving the resulting model. It is the ultimate goal of business analytics to solve any business problem in a rational data-driven way. Many of these problems, especially the strategic ones ("which product to develop?" or "how to maximize profit while keeping the risk of a loss below 5%?", or even "which employees to hire?") are very hard to model. Therefore modeling is an essential part of business analytics. Solving these types of problems in a rational mathematical way was the promise of management science in the 1950s. This failed and OR/MS was largely focused on operational problems in the following decades. The availability of data brings the solution to these problems within reach.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 7, "Combinatorial Optimization," §7.4.
