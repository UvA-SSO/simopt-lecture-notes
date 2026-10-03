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
# description: "What an algorithm is, how we write one down in this lecture, and the three ways to characterize it: dedicated or general, polynomial or not, exact or heuristic."
# thumbnail: null
# ---
# # Lecture 11: Algorithms and Their Characteristics
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture11_algorithms.ipynb)

# %% [markdown]
# In Lectures 8 to 10 we modeled problems as (I)LO models and let a solver find the
# optimal solution. The solver runs an algorithm, such as the simplex method or branch
# and bound, but we did not look inside. This lecture does. This notebook explains
# what an algorithm is, how the algorithms in this lecture are written down, and how
# algorithms differ from each other. The next three notebooks each take one classical
# problem, give an (I)LO model for it and then a dedicated algorithm:
# [Shortest Path](lecture11_shortest-path.ipynb),
# [Maximum Flow](lecture11_maximum-flow.ipynb) and the
# [Traveling Salesman Problem](lecture11_tsp-heuristics.ipynb), where we also meet
# heuristics. The last notebook, [Complexity](lecture11_complexity.ipynb), explains
# why some of these problems can be solved fast and others apparently cannot.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - explain what an algorithm is and give examples;
# - read an algorithm written with a start step, a repeated step and a stopping rule,
#   and say what each intermediate quantity means;
# - characterize an algorithm as dedicated or general, as polynomial or
#   non-polynomial, and as exact or heuristic.

# %% [markdown]
# ## What Is an Algorithm?
#
# An **algorithm** is a set of computer instructions that finds a solution for a
# problem. More precisely, it is a computational procedure that yields a solution for
# every problem in a class of problems: not for one specific instance, but for every
# instance of the same type. Examples we have already met or will meet in this
# lecture:
#
# - the simplex method solves LO problems;
# - the branch-and-bound method solves ILO problems (see
#   [Integer Optimization](lecture8_integer-optimization.ipynb));
# - Dijkstra's algorithm solves the shortest path problem (see
#   [Shortest Path](lecture11_shortest-path.ipynb)).

# %% [markdown]
# (algorithm-style)=
# ### How We Write Down an Algorithm
#
# Most algorithms in this lecture improve something step by step: a distance
# estimate, a flow, a tour. To make clear what is going on halfway through, we first
# define the quantities the algorithm keeps track of and say exactly what they mean
# at every step. The algorithm itself then has three parts:
#
# - **Start:** give every quantity its initial value;
# - **While** some condition holds: the steps that are repeated, each one updating
#   the quantities;
# - the stopping rule (the condition of the while loop fails) and what the
#   quantities then give us.
#
# A small example shows the style. We want the largest of $n$ numbers
# $a_1, a_2, \dots, a_n$. Define, for $k = 1, \dots, n$,
#
# $$
# m_k = \text{the largest of the first } k \text{ numbers } a_1, \dots, a_k.
# $$
#
# Our aim is $m_n$. The algorithm is:
#
# - **Start:** $k = 1$ and $m_1 = a_1$.
# - **While** $k < n$:
#   - $m_{k+1} = \max\{m_k, a_{k+1}\}$;
#   - $k = k + 1$.
#
# The update step is correct because of what $m_k$ means: the largest of the first
# $k + 1$ numbers is either the largest of the first $k$ numbers, or the new number
# $a_{k+1}$. When the loop stops, $k = n$ and $m_n$ is the answer. In Python:

# %%
numbers = [4, 9, 2, 7, 11, 3]

largest = numbers[0]  # m_1
for a in numbers[1:]:
    largest = max(largest, a)  # m_{k+1} = max(m_k, a_{k+1})
print("largest number:", largest)

# %% [markdown]
# The algorithm makes $n - 1$ comparisons, one per repetition of the loop, so its
# running time grows linearly with $n$. Counting steps like this is the start of
# [Complexity](lecture11_complexity.ipynb).
#
# Dijkstra's algorithm in the next notebook is written in the same way, with
# $d_W(x)$, the length of a shortest path to node $x$ that only passes through nodes
# in a set $W$, in the role of $m_k$.

# %% [markdown]
# ## Characteristics of Algorithms
#
# Algorithms can be characterized in three ways.
#
# - **Dedicated or general.** A dedicated algorithm solves one type of problem, such
#   as Dijkstra's algorithm for the shortest path problem or the Ford-Fulkerson
#   algorithm for the maximum flow problem. A general algorithm solves a wide range
#   of problems: the simplex method solves any LO problem, branch and bound any ILO
#   problem, and evolutionary algorithms can be applied to almost any optimization
#   problem. A dedicated algorithm uses the structure of its problem, which usually
#   makes it much faster than solving an (I)LO model of the same problem with a
#   general algorithm.
# - **Polynomial or non-polynomial complexity.** The complexity of an algorithm
#   describes how its running time grows with the size of the problem, for example
#   with the number of nodes $n$ of a network. If the running time grows like $n^2$
#   (polynomial), large instances are no problem; if it grows like $n!$
#   (non-polynomial), even moderate instances take forever. See
#   [Complexity](lecture11_complexity.ipynb).
# - **Exact or heuristic.** An exact algorithm is guaranteed to find an optimal
#   solution. A **heuristic** is a best-effort method: a shortcut algorithm whose aim
#   is not the optimal solution but a good solution in a reasonable amount of time.
#   Training a neural network by gradient descent is an example: it finds good
#   network weights, but nobody claims they are the best possible ones.
#
# Why not just use the best general algorithm for everything? The **no free lunch
# theorem** (Wolpert & Macready, 1997) says that no such best algorithm exists:
# averaged over all possible problems, all search and optimization algorithms
# perform equally well. If an algorithm does better on one class of problems, it pays
# for that with worse performance on another class. Good performance comes from
# matching the algorithm to the structure of the problem, which is what dedicated
# algorithms do.
#
# The table below characterizes the algorithms of Lectures 8 to 11. The running times
# are explained in the notebooks of the algorithms and in
# [Complexity](lecture11_complexity.ipynb).
#
# :::{table} Characteristics of the algorithms in Lectures 8 to 11.
# :label: tbl-algorithm-characteristics
#
# | algorithm | problem | dedicated or general | complexity | exact or heuristic |
# |---|---|---|---|---|
# | simplex method | LO | general | non-polynomial in the worst case, fast in practice | exact |
# | branch and bound | ILO | general | non-polynomial | exact |
# | Dijkstra's algorithm | shortest path | dedicated | polynomial ($n^2$) | exact |
# | Ford-Fulkerson algorithm | maximum flow | dedicated | polynomial, if augmenting paths are chosen well | exact |
# | brute force (try all tours) | TSP | dedicated | non-polynomial ($n!$) | exact |
# | 2-opt | TSP | dedicated | fast per improvement step | heuristic |
# :::

# %% [markdown]
# (graph-notation)=
# ## Graphs: Notation for This Lecture
#
# The three problems in this lecture are defined on a
# [graph](lecture8_linear-optimization.ipynb#graphs-intro): a set of nodes $V$
# connected by arcs. An arc from node $i$ to node $j$ is written $i \to j$ or
# $(i, j)$ and has a number $c_{ij}$: a distance in the shortest path problem, a
# capacity in the maximum flow problem. An undirected edge between $i$ and $j$, which
# can be used in both directions, is the same as two arcs $i \to j$ and $j \to i$
# with the same number. The number of nodes is $n = |V|$.
#
# In Python we store a graph as a dictionary keyed by arcs, the same way the
# transportation costs were stored in
# [Transportation and Transshipment](lecture9_transportation.ipynb):

# %%
distance = {("A", "B"): 2, ("A", "C"): 1, ("C", "B"): 2}
print("distance from A to B:", distance["A", "B"])

# %% [markdown]
# A pair of nodes without an arc simply has no key. In the mathematical models we
# then set $c_{ij} = \infty$ for a distance (the arc is never worth using) or
# $c_{ij} = 0$ for a capacity (nothing can flow through it).

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 7,
#   "Combinatorial Optimization," introduction.
# - Wolpert, D. H., & Macready, W. G. (1997). No free lunch theorems for
#   optimization. *IEEE Transactions on Evolutionary Computation*, 1(1), 67-82.
