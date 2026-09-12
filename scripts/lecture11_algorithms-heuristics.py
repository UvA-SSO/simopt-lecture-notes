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
# # Lecture 11: Algorithms and Heuristics

# %% [markdown]
# Combinatorics is the mathematical study of finite structures. Counting problems (e.g., how many ways are there to select $k$ items out of $n$?) are a good example of a combinatorial problem. Combinatorial optimization (CO) considers optimization problems over these finite structures. Usually the number of elements is too big for the problem to be solved by enumeration. Therefore algorithms are at the heart of CO.
#
# Integer linear optimization problems (see [Integer Optimization](lecture8_integer-optimization.ipynb)) with all variables integer have finite (or countable) feasible regions, but they are usually not considered to be part of CO. Together they constitute discrete optimization. As we will see, many CO problems have dedicated algorithms but can also be formulated as ILO problems. In the next sections we deal with a few of the most common CO problems. A common theme is complexity, which will be addressed separately (see [Complexity](lecture11_complexity.ipynb)).
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - solve certain combinatorial problems, especially the shortest path, the maximum flow and the traveling salesman problems
# - describe the concept of complexity and reflect on its implications for algorithms and problem solving

# %% [markdown]
# ## Algorithms and Their Characteristics
#
# An algorithm is a computational procedure that produces a solution for a class of problems: the simplex method for LO, branch-and-bound for ILO, Dijkstra's algorithm for shortest path — all are algorithms. It is useful to characterize an algorithm along three axes:
#
# - **Dedicated or general.** Dijkstra's algorithm and the Ford-Fulkerson algorithm below are dedicated to one specific problem structure; the simplex method, branch-and-bound, and evolutionary algorithms are general-purpose. A dedicated algorithm exploits the problem structure and is typically much faster than a general one — this is precisely why it is worth studying Dijkstra's algorithm at all, even though the shortest path problem can already be formulated as an LO problem (see below).
# - **Polynomial or non-polynomial complexity**, i.e., how the running time grows with the size of the problem — the topic of the [Complexity](lecture11_complexity.ipynb) notebook.
# - **Exact or heuristic.** An exact algorithm is guaranteed to find the optimum; a heuristic is not, but aims to find a good solution in a reasonable amount of time instead.
#
# A well-known folklore result, the "no free lunch" theorem, is a useful caution here: informally, no search algorithm can be uniformly better than every other algorithm across *all* possible problems — any advantage gained on one class of problems is paid for by a disadvantage on another. There is no single best algorithm, only algorithms well- or poorly-suited to a given problem structure.
#
# The rest of this notebook demonstrates three classical combinatorial problems, each with (i) an (I)LO formulation and (ii) a dedicated algorithm: shortest path, maximum flow, and the traveling salesman problem.

# %% [markdown]
# ## The Shortest Path Problem
#
# The shortest path problem consists of finding the shortest distance between two nodes in a (uni or bi-directional) [graph](lecture8_linear-optimization.ipynb#graphs-intro) with (positive) distances on the arcs. The importance of this problem is evident: it has to be solved every time we use navigation software, and many other less obvious applications exist.
#
# The best-known algorithm was invented by the Dutch computer scientist Edsger Dijkstra (1930-2002). His algorithm works as follows: let $V$ be the set of nodes in the graph, and $E$ the edges connecting them, and $c_{ij}$ the distance from $i$ to $j$. See the figure below for an example. Let $s$ be the node from which we want to compute the shortest routes (Dijkstra calculates the shortest route to all nodes at the same time). The algorithm works as follows:
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
# As an example let us determine the shortest path from A to F in [](#fig-shortest-path-graph). When an edge is not present the distance is $\infty$. In [](#tbl-dijkstra) we see the algorithm step by step. Underlined numbers correspond to visited nodes.

# %% [markdown]
# :::{figure} images/lecture11_fig7.1.png
# :label: fig-shortest-path-graph
#
# A graph with V = {A, B, C, D, E, F} and distances along the edges.
# :::

# %% [markdown]
# :::{table} Dijkstra's algorithm applied to the graph above (each column shows the running distance estimate; blank means unchanged).
# :label: tbl-dijkstra
#
# | step | A | B | C | D | E | F |
# |---|---|---|---|---|---|---|
# | 1 | 0 | ∞ | ∞ | ∞ | ∞ | ∞ |
# | 2 | | 3 | 1 | ∞ | ∞ | ∞ |
# | 3 | | 2 | | 3 | 5 | ∞ |
# | 4 | | | | 3 | 5 | ∞ |
# | 5 | | | | | 4 | 8 |
# | 6 | | | | | | 6 |
# :::
#
# We can check this by hand-worked result against pulp's LO formulation for the shortest path (see the box below), using the same graph data:

# %%
import pulp

distance = {
    ("A", "B"): 3,
    ("A", "C"): 1,
    ("C", "B"): 1,
    ("C", "D"): 2,
    ("C", "E"): 4,
    ("D", "E"): 1,
    ("D", "F"): 5,
    ("E", "F"): 2,
}
nodes = ["A", "B", "C", "D", "E", "F"]
source, destination = "A", "F"
BIG = 1e6  # stands in for infinity for non-existent arcs


def dist(i: str, j: str) -> float:
    return distance.get((i, j), distance.get((j, i), BIG))


shortest_path = pulp.LpProblem(name="shortest_path", sense=pulp.LpMinimize)
flow = {
    (i, j): pulp.LpVariable(name=f"x_{i}_{j}", lowBound=0) for i in nodes for j in nodes if i != j
}
shortest_path += pulp.lpSum(
    dist(i, j) * flow[i, j]
    for i in nodes
    for j in nodes
    if i != j and j != source and i != destination
)
shortest_path += pulp.lpSum(flow[source, j] for j in nodes if j != source) == 1
for k in nodes:
    if k in (source, destination):
        continue
    shortest_path += pulp.lpSum(
        flow[i, k] for i in nodes if i != k and i != destination
    ) == pulp.lpSum(flow[k, j] for j in nodes if j != k and j != source)

shortest_path.solve(pulp.PULP_CBC_CMD(msg=False))
print("shortest distance A to F:", shortest_path.objective.value())

# %% [markdown]
# matching Dijkstra's own result of 6 from [](#tbl-dijkstra) above.

# %% [markdown]
# This algorithm terminates with $f(x)$ the shortest distance from $s$ to $j$ for any node $j$. Every time you update a node you can keep track of the minimizing arc. If you store this arc every time a node becomes visited, then you build a tree with the shortest paths to all nodes.
#
# :::{exercise}
# :label: ex-7-1
#
# Find the shortest path from A to E for [](#fig-shortest-path-exercise) using Dijkstra's algorithm. Formulate it also as an LO problem using the formulation in the box below, solve it with pulp, and check that the two answers agree.
# :::

# %% [markdown]
# :::{figure} images/lecture11_fig7.2.png
# :label: fig-shortest-path-exercise
#
# A directed graph with V = {A, B, C, D, E} and distances along the arcs.
# :::

# %% [markdown]
# :::{note} LO Solution of the Shortest Path Problem
# The shortest path problem can also be formulated as an LO problem. It is a special case of the [transshipment problem](lecture9_ilo-applications.ipynb#transshipment-problem), with demand 1. We determine the shortest path from the source $s$ to a single destination $d$. The LO formulation is as follows:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i \ne d} \sum_{j \ne s} c_{ij} x_{ij} \\
# \text{subject to} \quad & \sum_{j \ne s} x_{sj} = 1; \\
# & \sum_{i \ne d} x_{ik} = \sum_{j \ne s} x_{kj} \text{ for all intermediate nodes } k \ne s \text{ and } k \ne d; \\
# & x_{ij} \ge 0 \text{ for all } i \ne j.
# \end{aligned}
# $$
#
# Because there is only one destination $d$, the flow into it must be 1 and there is no need for a separate constraint. It can happen that the solution is non-integer, that the flow is split between multiple paths from $s$ to $d$. In that case, every path with positive flow is optimal.
#
# If we assume $c_{ij} = \infty$ if an arc does not exist then all summations can range over $V$.
#
# Note that this formulation has $n^2$ variables and $n$ constraints. Although LO problems can be solved efficiently, Dijkstra's algorithm is much faster — which is exactly why pulp took noticeably more setup above (defining a variable for every ordered pair of nodes) even though it reached the same answer as the one-pass Dijkstra table.
# :::
#
# It is interesting to look at the number of operations needed by Dijkstra, with $n = |V|$ the number of nodes. In every iteration of the algorithm, one node is made current; that makes $n$ iterations. In every iteration all non-visited nodes are updated, that makes between $n-1$ and 1 operations, depending on the iteration. Thus, the total number of operations $N(n) = c((n-1) + (n-2) + \dots + 1) = cn(n-1)/2$, with $c$ the (constant) number of operations to update one node. This is a second degree polynomial in $n$, therefore we say that $N$ is in the order of $n^2$, written $N(n) = O(n^2)$. Formally, this means that $\lim_{n \to \infty} N(n)/n^2$ is a constant.
#
# :::{exercise}
# :label: ex-7-2
#
# Implement the LO formulation of the shortest path problem in pulp for the problem of [](#fig-shortest-path-graph), following the pattern used above.
# :::
#
# :::{note} Navigation Software
# Dijkstra is essentially also the algorithm that is used in the software that is implemented in your TomTom software or in Apple or Google maps on your smartphone. However, when you calculate the shortest route from A(msterdam) to B(erlin), Dijkstra also computes roads to all other possible destinations which is highly inefficient.
#
# ![Illustration of a shortest-route calculation from Amsterdam to Berlin](images/lecture11_box7.2-map.png)
#
# Several solutions have been proposed in the literature to tackle this problem. We discuss a well-known one: the A* algorithm. It adapts Dijkstra's algorithm in the following way. An evaluation function $e(i)$ is introduced with $e(i) = f(i) + h(i)$, where $f(i)$ is the current distance from the origin to $i$ and $h(i)$ is an estimation of the distance from $i$ to the destination, for example based on the Euclidian distance (based on a straight line instead of the road network) and the maximum highway speed. Now the node with smallest value of $e(i)$ is made current. Under certain conditions ($h(i)$ must be smaller than the real distance) this new algorithm finds again the optimal route, and it is dramatically faster than Dijkstra.
# :::

# %% [markdown]
# ## The Maximum Flow Problem
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
# It needs to be specified how such an augmenting path can be found and how to determine and add its capacity. There are multiple ways to do this. We will not go into the technical details, instead we illustrate this graphically through an example which can be found as the upper-left graph below. The source is A and the destination is F. Initially all flows are 0, thus the first augmenting path is simply a path in the graph from A to F, let's say A-B-D-F. The minimum capacity on the arcs is 2 (A-B), thus the capacity of the augmenting path is 2. We add this to the flow, leading to the upper-right graph below.
#
# We try to find a new augmenting path. We start from A. B cannot be reached, because its capacity is already fully used. C can be reached, from which we can reach B, D and E. From D we can reach F. This leads to the tree of the bottom-left graph, from which we derive the augmenting path A-C-D-F, with capacity 2 (because only 2 is left on D-F). After 2 more iterations we find the flow of the lower-right graph with value 6. When we try to find a new augmenting path we cannot reach E and F. Indeed, the sum of the capacities of the arcs from $\{A, B, C, D\}$ to $\{E, F\}$, called a cut, has value 6. There is a well-known theorem that states that the minimum over all cuts is equal to the maximum flow. Because we found a flow and a cut of 6 we are sure to have found an optimal solution.

# %% [markdown]
# :::{figure} images/lecture11_fig7.3.png
# :label: fig-max-flow
#
# Ford-Fulkerson illustrated.
# :::
#
# > **Erratum applied (p. 112):** the second graph (upper-right, after the first augmenting path A-B-D-F) illustrates a flow of 2 instead of 5 — consistent with the text, which adds the augmenting-path capacity 2 to the flow.

# %% [markdown]
# :::{note} LO Solution of the Maximum Flow Problem
# Also the max flow problem can be formulated as LO problem. The objective is to maximize the flow out of the source $s$ such that capacity constraints are not violated and all intermediate nodes have flow in = flow out.
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_{j \ne s} x_{sj} \\
# \text{subject to} \quad & x_{ij} \le c_{ij} \text{ for all } i, j,\ i \ne j; \\
# & \sum_{i \ne d} x_{ik} = \sum_{j \ne s} x_{kj} \text{ for all intermediate nodes } k \ne s \text{ and } k \ne d; \\
# & x_{ij} \ge 0 \text{ for all } i, j,\ i \ne j.
# \end{aligned}
# $$
#
# Note that there are $n(n-1)$ variables and $2n(n-1) + n - 2$ constraints, with $n$ the number of nodes in the network. Note also that these are 2nd order polynomials in $n$.
# :::
#
# Let's check this LO formulation against the Ford-Fulkerson result above, using the capacities from [](#fig-max-flow):

# %%
capacity = {
    ("A", "B"): 2,
    ("A", "C"): 5,
    ("B", "D"): 3,
    ("C", "B"): 2,
    ("C", "D"): 3,
    ("C", "E"): 1,
    ("D", "E"): 1,
    ("D", "F"): 4,
    ("E", "F"): 2,
}
nodes = ["A", "B", "C", "D", "E", "F"]
source, sink = "A", "F"

max_flow = pulp.LpProblem(name="max_flow", sense=pulp.LpMaximize)
flow = {
    (i, j): pulp.LpVariable(name=f"x_{i}_{j}", lowBound=0, upBound=cap)
    for (i, j), cap in capacity.items()
}
max_flow += pulp.lpSum(flow[source, j] for j in nodes if (source, j) in capacity)
for k in nodes:
    if k in (source, sink):
        continue
    inflow = pulp.lpSum(flow[i, k] for i in nodes if (i, k) in capacity)
    outflow = pulp.lpSum(flow[k, j] for j in nodes if (k, j) in capacity)
    max_flow += inflow == outflow

max_flow.solve(pulp.PULP_CBC_CMD(msg=False))
print("maximum flow:", max_flow.objective.value())

# %% [markdown]
# again 6, matching the flow (and the cut) found by hand above.
#
# :::{exercise}
# :label: ex-7-3
#
# Determine the maximum flow from A to H in [](#fig-max-flow-exercise). Check that there is a cut with the same value. Determine the optimal solution by implementing the LO solution above in pulp.
# :::

# %% [markdown]
# ## The Traveling Salesman Problem
#
# The third famous problem we consider is the traveling salesman problem or TSP. For an undirected graph with distances on the edges the objective is to visit all the nodes or cities in a closed tour with minimal total distance without visiting the same node twice. For example, if the numbers in [](#fig-max-flow-exercise) are interpreted as distances, then the tour that visits the cities in a clockwise or counter-clockwise manner has distance 21. See [](#fig-2opt-before). However, it is possible to find shorter tours. Note that it doesn't matter which node is taken as a starting point because we have to return to that point anyway.

# %% [markdown]
# :::{figure} images/lecture11_fig7.4.png
# :label: fig-max-flow-exercise
#
# An undirected graph with capacities.
# :::
#
# > **Erratum applied (p. 113):** the numbers along the edges of this graph are distances, not capacities (the maximum-flow exercise above uses them as capacities, while the TSP discussion below uses the same numbers as distances).

# %% [markdown]
# Just like shortest path and max flow, the TSP has an ILO formulation. Let $x_{ij} \in \{0, 1\}$ indicate whether edge $ij$ is used in the tour, and $c_{ij}$ the distance between $i$ and $j$ (with $c_{ii}$ set very large so that no self-loop is ever chosen). The natural formulation is:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i,j,\, i \ne j} c_{ij} x_{ij} \\
# \text{subject to} \quad & \sum_{i \ne k} x_{ik} = \sum_{j \ne k} x_{kj} = 1 \text{ for all nodes } k; \\
# & x_{ij} \in \{0, 1\} \text{ for all } i, j,\ i \ne j.
# \end{aligned}
# $$
#
# Requiring exactly one incoming and one outgoing edge per node, however, does not by itself force a single tour through all $n$ nodes — it is equally satisfied by several disjoint sub-tours that together cover every node (e.g., two separate 3-node tours in a 6-node graph). To rule this out we add a **subtour elimination constraint** for every subset $S$ of nodes with $2 \le |S| \le n-2$: a tour restricted to the nodes in $S$ alone would use exactly $|S|$ edges, so forcing
#
# $$
# \sum_{i,j \in S} x_{ij} \le |S| - 1 \quad \text{for all } S \subset V \text{ with } 2 \le |S| \le n-2
# $$
#
# makes any such subtour infeasible. This repairs the model, but at a cost: an $n$-node graph has $2^n$ subsets, so this adds an exponential number of constraints — impractical to write down explicitly for anything but tiny instances. (In practice, solvers add these constraints lazily: solve the relaxed model, and if the solution contains a subtour, add just that subtour's constraint and re-solve, repeating until a single tour emerges.)
#
# For the TSP, no efficient exact algorithm is known — not via this ILO formulation, nor via any other approach; essentially only enumeration is guaranteed to find the shortest tour. However, there are $(n-1)!$ different tours, which will take very long even for moderately sized problems! Therefore the TSP is in practice solved using heuristics, i.e., algorithms that are not guaranteed to terminate with an optimal solution.
#
# ### Heuristics
#
# Because no fast exact algorithm exists for the TSP — or, more generally, for any [NP-complete](lecture11_complexity.ipynb) problem — we turn to heuristics: procedures that aim for a good solution in a reasonable amount of time rather than a guaranteed optimum. Heuristics come in two flavors:
#
# - **Metaheuristics** are not tailored to any one specific problem: local search, evolutionary algorithms, tabu search, and simulated annealing all fall in this category, and can in principle be pointed at any optimization problem (though some problem-specific tailoring, e.g. how a "move" is defined, is usually still needed).
# - **Problem-specific heuristics** exploit the structure of one particular problem. The 2-opt heuristic below is a classic example for the TSP; most well-studied combinatorial problems have their own body of literature of dedicated heuristics.
#
# An example of a problem-specific heuristic — and a form of local search — is 2-opt. It starts with some initial tour and then one by one all combinations of 2 edges are removed from the tour and replaced by the other 2 edges connecting them in order to make a tour again. When an edge does not exist, we assume it has length $\infty$. When the length of the new tour is shorter, we consider it as our next solution. This continues until no improvement can be found.
#
# Let us apply this to the graph above. We start with the left tour below which has length 21. If we replace the edges BD and EG we get the right graph below with length 18. No further improvements can be found.

# %% [markdown]
# :::{figure} images/lecture11_fig7.5-left.png
# :label: fig-2opt-before
#
# Initial tour, length 21.
# :::
#
# :::{figure} images/lecture11_fig7.5-right.png
# :label: fig-2opt-after
#
# After replacing edges BD and EG, length 18 — an illustration of the 2-opt heuristic.
# :::

# %% [markdown]
# (local-search-heuristic)=
# The final tour of the algorithm is called a local optimum, because in the neighborhood of the final tour there is no better solution, but there is no guarantee that the solution is optimal. An overall best solution (there can be more than one) is called a global optimum. [](#fig-2opt-after) is a local optimum with respect to the 2-opt heuristic. It is also the global optimum.
#
# Other heuristics than 2-opt exist, for example 3-opt, which evidently consists of removing 3 edges from a tour and trying all the ways of reconnecting to make a tour again. These heuristics are also called local search methods: from a possible solution a neighborhood of solutions is defined which are searched. When a better one is found then it replaces the current one and the search continues from the new solution until no improvement can be made anymore. In the case of 2-opt the neighborhood of a tour consists of all tours that can be constructed from omitting 2 edges and adding the 2 edges which make the tour complete again. The quality of the local optimum depends strongly on the choice of neighborhood. For example, 3-opt is known to give better solutions than 2-opt.
#
# :::{exercise}
# :label: ex-7-4
#
# Construct a local optimum using 2-opt for [](#fig-tsp-distances) starting with the tour ABDFEC.
# :::

# %% [markdown]
# :::{figure} images/lecture11_fig7.6.png
# :label: fig-tsp-distances
#
# An undirected graph with distances.
# :::

# %% [markdown]
# The Python package [`python-tsp`](https://github.com/fillipe-gsm/python-tsp) exists by which we can solve TSP problems, offering both exact (dynamic programming) and heuristic (2-opt and others) solvers.
#
# :::{exercise}
# :label: ex-7-5
#
# Install `python-tsp`, read its documentation and use it to solve the problem of [](#fig-tsp-distances).
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. Chapter 7, "Combinatorial Optimization," §7.1–§7.3.
