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
# Lecture 11: Algorithms and Heuristics; Complexity

# %% [markdown]
# ## Source map
# - **Schedule topic(s):** Algorithms and heuristics (§7.1-§7.3); Complexity (§7.4)
# - **Book source(s):** Koole, *An Introduction to Business Analytics* (2019) — Chapter 7 "Combinatorial Optimization", §7.1-§7.4
# - **Errata applied:** Table 7.1 (Dijkstra's algorithm applied to the graph of Figure 7.1) corrected per `Literature/Erratum Table 7.1 book Ger Koole.xlsx`
#
# > **Uncertainty:** the book's own chapter/section title is "Combinatorial Optimization" (§7.1 shortest path, §7.2 maximum flow, §7.3 traveling salesman, §7.4 complexity), which differs from the schedule's label "Algorithms and heuristics" / "Complexity" — the section numbers map 1:1, only the descriptive label differs.

# %% [markdown]
# ## Compiled source text

# %% [markdown]
# ### Chapter 7 — Combinatorial Optimization
#
# Combinatorics is the mathematical study of finite structures. Counting problems (e.g., how many ways are there to select $k$ items out of $n$?) are a good example of a combinatorial problem. Combinatorial optimization (CO) considers optimization problems over these finite structures. Usually the number of elements is too big for the problem to be solved by enumeration. Therefore algorithms are at the heart of CO.
#
# Integer linear optimization problems (see Chapter 6) with all variables integer have finite (or countable) feasible regions, but they are usually not considered to be part of CO. Together they constitute discrete optimization. As we will see, many CO problems have dedicated algorithms but can also be formulated as ILO problems. In the next sections we deal with a few of the most common CO problems. A common theme is complexity, which will be addressed in a separate section.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - solve certain combinatorial problems, especially the shortest path, the maximum flow and the traveling salesman problems
# - describe the concept of complexity and reflect on its implications for algorithms and problem solving

# %% [markdown]
# #### 7.1 The shortest path problem
#
# The shortest path problem consists of finding the shortest distance between two nodes in a (uni or bi-directional) graph (see page 92) with (positive) distances on the arcs. The importance of this problem is evident: it has to be solved every time we use navigation software, and many other less obvious applications exist.
#
# The best-known algorithm was invented by the Dutch computer scientist Edsger Dijkstra (1930-2002). His algorithm works as follows: let $V$ be the set of nodes in the graph, and $E$ the edges connecting them, and $c_{ij}$ the distance from $i$ to $j$. See Figure 7.1 for an example. Let $s$ be the node from which we want to compute the shortest routes (Dijkstra calculates the shortest route to all nodes at the same time). The algorithm works as follows:
#
# ```
# initialization: f(s) = 0, f(i) = ∞ for all i ∈ V\{s},
#                 s is visited and current
# repeat until all nodes are visited:
#     update all unvisited neighbors j of current node i, i.e., update
#         f(j) to f(i) + c_ij if f(i) + c_ij < f(j)
#     determine unvisited node with smallest value
#     make it visited and current
# ```
#
# As an example let us determine the shortest path from A to F in the graph of Figure 7.1. When an edge is not present the distance is $\infty$. In Table 7.1 we see the algorithm step by step. Underlined numbers correspond to visited nodes.

# %% [markdown]
# ![Figure 7.1: A graph with V = {A, B, C, D, E, F} and distances along the edges](images/lecture11_fig7.1.png)

# %% [markdown]
# | step | A | B | C | D | E | F |
# |---|---|---|---|---|---|---|
# | 1 | 0 (–) | ∞ | ∞ | ∞ | ∞ | ∞ |
# | 2 | | 3 (A) | 1 (A) | ∞ | ∞ | ∞ |
# | 3 | | 2 (C) | | 3 (C) | 5 (C) | ∞ |
# | 4 | | | | 3 (C) | 5 (C) | ∞ |
# | 5 | | | | | 4 (D) | 8 (D) |
# | 6 | | | | | | 6 (E) |
#
# Table 7.1: Dijkstra's algorithm applied to the graph of Figure 7.1. Each cell shows the updated distance and, in parentheses, the node it was updated from; blank cells carry forward the previous value unchanged.
#
# > **Erratum applied:** Table 7.1 corrected per `Literature/Erratum Table 7.1 book Ger Koole.xlsx` — the corrected table adds, for each updated cell, the predecessor node the update came from (in parentheses), clarifying which arc produced each shortest-distance update.

# %% [markdown]
# ![Figure 7.2: A directed graph with V = {A, B, C, D, E} and distances along the arcs](images/lecture11_fig7.2.png)

# %% [markdown]
# > **Box 7.1. LO solution of shortest path problem**
# >
# > The shortest path problem can also be formulated as an LO problem. It is a special case of the transshipment problem of Section 6.3, with demand 1. We determine the shortest path from the source $s$ to a single destination $d$. The LO formulation is as follows:
# >
# > $$
# > \begin{aligned}
# > \text{minimize} \quad & \sum_{i \ne d} \sum_{j \ne s} c_{ij} x_{ij} \\
# > \text{subject to} \quad & \sum_{j \ne s} x_{sj} = 1; \\
# > & \sum_{i \ne d} x_{ik} = \sum_{j \ne s} x_{kj} \text{ for all intermediate nodes } k \ne s \text{ and } k \ne d; \\
# > & x_{ij} \ge 0 \text{ for all } i \ne j.
# > \end{aligned}
# > $$
# >
# > Because there is only one destination $d$, the flow into it must be 1 and there is no need for a separate constraint. It can happen that the solution is non-integer, that the flow is split between multiple paths from $s$ to $d$. In that case, every path with positive flow is optimal.
# >
# > If we assume $c_{ij} = \infty$ if an arc does not exist then all summations can range over $V$.
# >
# > Note that this formulation has $n^2$ variables and $n$ constraints. Although LO problems can be solved efficiently, Dijkstra's algorithm is much faster.
#
# It is interesting to look at the number of operations needed by Dijkstra, with $n = |V|$ the number of nodes. In every iteration of the algorithm, one node is made current; that makes $n$ iterations. In every iteration all non-visited nodes are updated, that makes between $n-1$ and 1 operations, depending on the iteration. Thus, the total number of operations $N(n) = c((n-1) + (n-2) + \dots + 1) = cn(n-1)/2$, with $c$ the (constant) number of operations to update one node. This is a second degree polynomial in $n$, therefore we say that $N$ is in the order of $n^2$, written $N(n) = O(n^2)$. Formally, this means that $\lim_{n \to \infty} N(n)/n^2$ is a constant.
#
# **Exercise 7.2** Implement the LO formulation of the shortest path problem in AMPL and use the NEOS server to solve the problem of Figure 7.1.
#
# > **Box 7.2. Navigation software**
# >
# > Dijkstra is essentially also the algorithm that is used in the software that is implemented in your TomTom software or in Apple or Google maps on your smartphone. However, when you calculate the shortest route from A(msterdam) to B(erlin), Dijkstra also computes roads to all other possible destinations which is highly inefficient.

# %% [markdown]
# ![Box 7.2: illustration of a shortest-route calculation from Amsterdam to Berlin](images/lecture11_box7.2-map.png)

# %% [markdown]
# > Several solutions have been proposed in the literature to tackle this problem. We discuss a well-known one: the A* algorithm. It adapts Dijkstra's algorithm in the following way. An evaluation function $e(i)$ is introduced with $e(i) = f(i) + h(i)$, where $f(i)$ is the current distance from the origin to $i$ and $h(i)$ is an estimation of the distance from $i$ to the destination, for example based on the Euclidian distance (based on a straight line instead of the road network) and the maximum highway speed. Now the node with smallest value of $e(i)$ is made current. Under certain conditions ($h(i)$ must be smaller than the real distance) this new algorithm finds again the optimal route, and it is dramatically faster than Dijkstra.

# %% [markdown]
# #### 7.2 The maximum flow problem
#
# A second classic CO problem is the maximum flow problem. In a (possibly directed) graph we add a capacity to every edge. One node is the source, another node is the destination. The goal of the max flow problem is to transport as much as possible from source to destination using the edges of the graph without exceeding the maximal capacity on any edge. The Ford-Fulkerson algorithm was invented to solve this problem. It works as follows:
#
# ```
# initialization: assign flow of 0 to all arcs
# repeat:
#     find an augmenting path
#     add capacity of augmenting path to flow
# until no more augmenting path can be found
# ```
#
# It needs to be specified how such an augmenting path can be found and how to determine and add its capacity. There are multiple ways to do this. We will not go into the technical details, instead we illustrate this graphically through an example which can be found as the upper-left graph in Figure 7.3. The source is A and the destination is F. Initially all flows are 0, thus the first augmenting path is simply a path in the graph from A to F, let's say A-B-D-F. The minimum capacity on the arcs is 2 (A-B), thus the capacity of the augmenting path is 2. We add this to the flow, leading to the upper-right graph of Figure 7.3.
#
# We try to find a new augmenting path. We start from A. B cannot be reached, because its capacity is already fully used. C can be reached, from which we can reach B, D and E. From D we can reach F. This leads to the tree of the bottom-left graph, from which we derive the augmenting path A-C-D-F, with capacity 2 (because only 2 is left on D-F). After 2 more iterations we find the flow of the lower-right graph with value 6. When we try to find a new augmenting path we cannot reach E and F. Indeed, the sum of the capacities of the arcs from $\{A, B, C, D\}$ to $\{E, F\}$, called a cut, has value 6. There is a well-known theorem that states that the minimum over all cuts is equal to the maximum flow. Because we found a flow and a cut of 6 we are sure to have found an optimal solution.
#
# **Exercise 7.3** Determine the maximum flow from A to H in the undirected graph of Figure 7.4. Check that there is a cut with the same value. Determine the optimal solution by implementing the LO solution in AMPL and/or Excel.

# %% [markdown]
# ![Figure 7.3: Ford-Fulkerson illustrated](images/lecture11_fig7.3.png)

# %% [markdown]
# > **Box 7.3. LO solution of maximum flow problem**
# >
# > Also the max flow problem can be formulated as LO problem. The objective is to maximize the flow out of the source $s$ such that capacity constraints are not violated and all intermediate nodes have flow in = flow out.
# >
# > $$
# > \begin{aligned}
# > \text{maximize} \quad & \sum_{j \ne s} x_{sj} \\
# > \text{subject to} \quad & x_{ij} \le c_{ij} \text{ for all } i, j,\ i \ne j; \\
# > & \sum_{i \ne d} x_{ik} = \sum_{j \ne s} x_{kj} \text{ for all intermediate nodes } k \ne s \text{ and } k \ne d; \\
# > & x_{ij} \ge 0 \text{ for all } i, j,\ i \ne j.
# > \end{aligned}
# > $$
# >
# > Note that there are $n(n-1)$ variables and $2n(n-1) + n - 2$ constraints, with $n$ the number of nodes in the network. Note also that these are 2nd order polynomials in $n$.

# %% [markdown]
# #### 7.3 The traveling salesman problem
#
# The third famous problem we consider is the traveling salesman problem or TSP. For an undirected graph with distances on the edges the objective is to visit all the nodes or cities in a closed tour with minimal total distance without visiting the same node twice. For example, if the numbers in Figure 7.4 are interpreted as distances, then the tour that visits the cities in a clockwise or counter-clockwise manner has distance 21. See the left graph of Figure 7.5. However, it is possible to find shorter tours. Note that it doesn't matter which node is taken as a starting point because we have to return to that point anyway.

# %% [markdown]
# ![Figure 7.4: An undirected graph with capacities](images/lecture11_fig7.4.png)

# %% [markdown]
# For the TSP, no efficient algorithm is known; essentially only enumeration is guaranteed to find the shortest tour. However, there are $(n-1)!$ different tours, which will take very long even for moderately sized problems! Therefore the TSP is solved using heuristics, i.e., algorithms that are not guaranteed to terminate with an optimal solution.
#
# An example of such a heuristic is 2-opt. It starts with some initial tour and then one by one all combinations of 2 edges are removed from the tour and replaced by the other 2 edges connecting them in order to make a tour again. When an edge does not exist, we assume it has length $\infty$. When the length of the new tour is shorter, we consider it as our next solution. This continues until no improvement can be found.
#
# Let us apply this to the graph of Figure 7.4. We start with the left tour of Figure 7.5 which has length 21. If we replace the edges BD and EG we get the right graph of Figure 7.5 with length 18. No further improvements can be found.

# %% [markdown]
# ![Figure 7.5 (left): initial tour, length 21](images/lecture11_fig7.5-left.png)
#
# ![Figure 7.5 (right): after replacing edges BD and EG, length 18](images/lecture11_fig7.5-right.png)
#
# Figure 7.5: An illustration of the 2-opt heuristic

# %% [markdown]
# The final tour of the algorithm is called a local optimum, because in the neighborhood of the final tour there is no better solution, but there is no guarantee that the solution is optimal. An overall best solution (there can be more than one) is called a global optimum. The right-hand graph is a local optimum with respect to the 2-opt heuristic. It is also the global optimum.
#
# Other heuristics than 2-opt exist, for example 3-opt, which evidently consists of removing 3 edges from a tour and trying all the ways of reconnecting to make a tour again. These heuristics are also called local search methods: from a possible solution a neighborhood of solutions is defined which are searched. When a better one is found then it replaces the current one and the search continues from the new solution until no improvement can be made anymore. In the case of 2-opt the neighborhood of a tour consists of all tours that can be constructed from omitting 2 edges and adding the 2 edges which make the tour complete again. The quality of the local optimum depends strongly on the choice of neighborhood. For example, 3-opt is known to give better solutions than 2-opt.
#
# **Exercise 7.4** Construct a local optimum using 2-opt for the graph of Figure 7.6 starting with the tour ABDFEC.

# %% [markdown]
# ![Figure 7.6: An undirected graph with distances](images/lecture11_fig7.6.png)

# %% [markdown]
# An R package "TSP" exists by which we can solve TSP problems.
#
# **Exercise 7.5** Install the R TSP package, read the manual and use it to solve the problem of Figure 7.4.

# %% [markdown]
# #### 7.4 Complexity
#
# In the previous sections we studied 3 archetypical CO problems: shortest path, max flow, and TSP. We saw a crucial difference between them: shortest path and max flow have run times polynomial in the size of the problem, but TSP grows as a factorial in $n$ (which is faster than exponential). This really makes a difference. The following table illustrates that.
#
# > **Box 7.4. ILO solution of the TSP**
# >
# > We can try to construct an LO formulation for the TSP, just like we did for the shortest path and the max flow problem. With $c_{ij}$ we denote the length of the edge between $i$ and $j$ and $x_{ij}$ is the binary variable which indicates whether the edge $ij$ is included in the tour. Then the obvious LO formulation is:
# >
# > $$
# > \begin{aligned}
# > \text{minimize} \quad & \sum_{i,j,\, i \ne j} c_{ij} x_{ij} \\
# > \text{subject to} \quad & \sum_{i \ne k} x_{ik} = \sum_{j \ne k} x_{kj} = 1 \text{ for all nodes } k; \\
# > & x_{ij} \in \{0, 1\} \text{ for all } i, j,\ i \ne j.
# > \end{aligned}
# > $$
# >
# > However, this problem we does not always result in a tour. For example, when applied to the graph of Figure 7.6, we get two disconnected tours with 3 nodes: ABC and DEF. Therefore we need additional constraints, for example requiring that any subset has fewer used edges than there are nodes in the subset:
# >
# > $$
# > \sum_{i,j \in S} x_{ij} \le |S| - 1 \text{ for all } S \subset V \text{ with } 2 \le |S| \le n-2.
# > $$
# >
# > By adding this constraint, we can solve the TSP to optimality. However, there are $2^n$ subsets of $V$, leading to an exponential number of constraints. This makes this solution approach practically infeasible.
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
# > **Box 7.5. Complexity of linear optimization**
# >
# > The simplex method, which is the original method invented by Dantzig for solving LO problems, has been proven to work very well, even for very big problems. However, it is possible to construct problems for which the run time is not polynomial in the size, thus the simplex method is not in P. In the 1980s, the so-called interior-point methods were developed that solve LO in polynomial time. These are widely used since then.
#
# It is important to note that there are more difficult classes of problems than NP-complete problems, for example those where it is hard to verify whether a proposed solution is indeed feasible. Note also that solving a mathematical optimization problem is only part of solving business problems. Indeed, translating a business problem into a mathematical model — modeling — is often harder than solving the resulting model. It is the ultimate goal of business analytics to solve any business problem in a rational data-driven way. Many of these problems, especially the strategic ones ("which product to develop?" or "how to maximize profit while keeping the risk of a loss below 5%?", or even "which employees to hire?") are very hard to model. Therefore modeling is an essential part of business analytics. Solving these types of problems in a rational mathematical way was the promise of management science in the 1950s. This failed and OR/MS was largely focused on operational problems in the following decades. The availability of data brings the solution to these problems within reach.
