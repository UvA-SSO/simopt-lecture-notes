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
# description: "Why some optimization problems (shortest path, maximum flow, LO) can be solved quickly while others (TSP, ILO in general) apparently cannot, and what that means in practice."
# thumbnail: null
# ---
# # Lecture 11: Complexity
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture11_complexity.ipynb)

# %% [markdown]
# In the previous notebooks we solved the
# [shortest path](lecture11_shortest-path.ipynb) and
# [maximum flow](lecture11_maximum-flow.ipynb) problems with fast exact algorithms,
# while for the [traveling salesman problem](lecture11_tsp-heuristics.ipynb) we only
# found an exact algorithm that tries all tours, and turned to heuristics. Complexity
# theory is about the running times of algorithms, and it explains this difference.
# It also tells us what to expect in practice: whether a solver will find the
# optimal solution of a large instance, or whether a heuristic is the better choice.
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
#   is not polynomial.

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
# - [trying all tours](lecture11_tsp-heuristics.ipynb) of a TSP means checking
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
# - **NP-complete** problems are problems for which no fast exact algorithm is
#   known. Large instances are solved with heuristics, or with solvers that may need
#   a very long time to prove optimality. Examples are the knapsack problem, machine
#   scheduling, set covering, the TSP and ILO in general.
#
# The name NP-complete needs some explanation. Formally, these classes are defined
# for yes-or-no questions, such as "is there a tour of length at most 16?". A
# question is in P if the answer can be computed in polynomial time. It is in NP
# if a "yes" can be checked in polynomial time once someone hands you a solution:
# given a tour, it is easy to check that it visits all cities and has length at most
# 16, even though finding such a tour may be hard. An NP-complete problem is a
# problem in NP that is at least as hard as every other problem in NP: every problem
# in NP can be translated into it in polynomial time. So if one NP-complete problem
# could be solved in polynomial time, all problems in NP could. An optimization
# problem is called NP-complete when its yes-or-no version is.
#
# For the modeling in this course, this means that adding integer variables to an LO
# model can change a problem that solves in a second into one that a solver cannot
# finish. [Solvers and How to Help Them](lecture10_solvers.ipynb) shows how to
# help a solver in that case.
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
# solved so far, by Grigori Perelman in 2003.

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
# :::{exercise}
# :label: ex-complexity-time
#
# A computer performs $10^9$ operations per second. For each of the running times
# $n^2$, $n^5$, $2^n$ and $n!$, what is the largest instance size $n$ that can be
# solved within one hour? Use Python to find out.
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 7,
#   "Combinatorial Optimization" (complexity).
