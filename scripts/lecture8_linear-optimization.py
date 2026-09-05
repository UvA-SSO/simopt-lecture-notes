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
# # Lecture 8: Linear Optimization

# %% [markdown]
# In this chapter we discuss linear optimization problems. It is a framework used successfully in many industries to solve a broad variety of problems. We discuss different types of linear problems and show how you can solve them in Python using [pulp](https://coin-or.github.io/pulp/), a Python library for linear and integer optimization.
#
# **Learning outcomes**
#
# On completion of this chapter, you will be able to:
#
# - implement linear optimization problems in Python
# - model appropriate business problems as linear optimization models
# - reflect on the usefulness and applicability of linear optimization

# %% [markdown]
# (problem-formulation)=
# ## Problem Formulation
#
# We introduce linear optimization through an example. Later on we will give the general formulation.
#
# Assume a company has $n$ products which it can produce using $m$ resources of which there is only a limited amount available. The problem to solve is which quantities to produce of each product such that the revenue is maximized and the resource constraints are satisfied. Examples of this *product-mix problem* are refineries combining different types of crude oil into end products, a farmer dividing his land between crops with constraints on the amount of fertilizer or environmental impact, etc.
#
# Let us consider a simple instance of this problem. A company has 2 differents products, for example 2 types of crops, with profit 2 and 3 per quantity produced. We also have 2 resources: fertilizer and land. Product 1 requires 1 unit (e.g., ton) of fertilizer and 1 unit (e.g., acre) of land per unit produced, product 2 requires only 2 units of land. The availability of the resources is as follows: 5 units of fertilizer, 10 units of land. What is the optimal product mix?
#
# This problem has a structure in which we can apply to many optimization problems:
#
# - there are decision variables that have to be chosen;
# - there is an objective that needs to be maximized;
# - there are constraints which need to be satisfied.
#
# In mathematical terms, our instance becomes:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 2x_1 + 3x_2 & \text{(objective)} \\
# \text{subject to} \quad & x_1 \le 5 & \text{(constraint resource 1)} \\
# & x_1 + 2x_2 \le 10 & \text{(constraint resource 2)} \\
# & x_1, x_2 \ge 0.
# \end{aligned}
# $$
#
# Because the functions $2x_1 + 3x_2$, $x_1$ and $x_1 + 2x_2$ are linear in $x = (x_1, x_2)$ we call this a linear optimization (LO) problem. For LO, efficient solvers exist that are guaranteed to give an optimal solution, even for problems with thousands of variables and constraints. To solve our instance with [pulp](https://coin-or.github.io/pulp/), a Python library for linear and integer optimization, we first import it and separate the problem data from the model:

# %%
import pulp

profit = [2, 3]  # profit per unit of product 1 and 2
resource_use = [[1, 0], [1, 2]]  # resource_use[i][j]: resource i needed per unit of product j
resource_availability = [5, 10]

# %% [markdown]
# Then we build the model: an `LpProblem`, one continuous decision variable per product, the objective, and one constraint per resource.

# %%
n_products = len(profit)
n_resources = len(resource_availability)

product_mix = pulp.LpProblem(name="product_mix", sense=pulp.LpMaximize)
x = [pulp.LpVariable(name=f"x_{j+1}", lowBound=0) for j in range(n_products)]

product_mix += pulp.lpSum(profit[j] * x[j] for j in range(n_products)), "profit"
for i in range(n_resources):
    product_mix += (
        pulp.lpSum(resource_use[i][j] * x[j] for j in range(n_products))
        <= resource_availability[i],
        f"resource_{i+1}",
    )

product_mix.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[product_mix.status])
print("optimal solution:", [xj.value() for xj in x])
print("optimal profit:", product_mix.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-1
#
# Re-solve the product-mix problem above by hand-deriving the corner points and checking which one pulp found.
# :::
#
# There are different ways to understand what pulp did to find this. Let us first take a graphical look. In [](#fig-lo-graphical) the problem is drawn with $x_1$ on the horizontal axis and $x_2$ on the vertical one. We see the two constraints who together with the non-negativity constraints ($x_1, x_2 \ge 0$, assumed by default) delimit the allowable area, often called the feasible region. Because of the linearity of the constraints and the objective, the optimum (if it exists, see below) must be at the edge, at a corner. To determine the optimal corner, we slide a line with equal objective value until we hit the feasible region, i.e., the set of points that satisfy all constraints. The line with value 36 is drawn in the figure. When we slide it down it hits the feasible region in the point with $x_1 = 5$ and $x_1 + 2x_2 = 10$. From this, the optimal solution follows again: $x_1 = 5$ and $x_2 = 2.5$, matching what pulp found above.

# %% [markdown]
# :::{figure} images/lecture8_fig6.1.png
# :label: fig-lo-graphical
#
# A graphical view of LO.
# :::
#
# > **Erratum applied (p. 87):** the objective line actually drawn in the figure corresponds to value 24, not 36 (the text above refers to the line "with value 36").

# %% [markdown]
# Let us now take an algebraic point of view. The 2 constraints can be rewritten as equalities as follows, using additional variables $y_1$ and $y_2$:
#
# $$
# \begin{aligned}
# x_1 \le 5 \\
# x_1 + 2x_2 \le 10 \\
# x_1, x_2 \ge 0
# \end{aligned}
# \quad\Leftrightarrow\quad
# \begin{aligned}
# x_1 + y_1 &= 5 \\
# x_1 + 2x_2 + y_2 &= 10 \\
# x_1, x_2, y_1, y_2 &\ge 0
# \end{aligned}
# $$
#
# We have 2 equalities and 4 variables. This means that 2 non-zero variables suffice to find a solution. All 4 corners of the feasible region correspond to such a solution. The interior corresponds to solutions for which all variables are positive. The corners correspond to the following solutions:
#
# - $(x_1, x_2, y_1, y_2) = (0, 0, 5, 10)$, origin;
# - $(x_1, x_2, y_1, y_2) = (0, 5, 5, 0)$, upper-left corner;
# - $(x_1, x_2, y_1, y_2) = (5, 0, 0, 5)$, lower-right corner;
# - $(x_1, x_2, y_1, y_2) = (5, 2.5, 0, 0)$, upper-right corner (the optimum).
#
# The algorithm implemented in the solver hops from corner to corner until it cannot improve the objective value anymore. This method is called the simplex algorithm.
#
# :::{note} History of Linear Optimization
# Several researchers have formulated Linear Optimization problems but it was G.B. Dantzig (1914-2005) who invented in 1947 the simplex algorithm. Dantzig was an American scientist with German-French roots. At the time, until very recently, it was commonly known as linear programming. LO has been extremely useful for solving all kinds of business problems and is by far the most successful technique within operations research.
#
# Its random, dynamic counterpart is called dynamic programming, developed by R.E. Bellman (1920-1984). This division between deterministic and random problems is still very visible in operations research theory and applications.
# :::
#
# The general formulation of an LO problem is as follows, for $n$ decision variables and $m$ constraints:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_{i=1}^{n} p_i x_i & \text{(objective)} \\
# \text{subject to} \quad & \sum_{j=1}^{n} a_{ij} x_j \le b_i, \quad i = 1, \dots, m & \text{(constraints)} \\
# & x_1, \dots, x_n \ge 0.
# \end{aligned}
# $$
#
# Instances where the objective needs to be minimized or with constraints of the form "=" or "≥" can be rewritten to fit the general formulation.
#
# :::{exercise}
# :label: ex-6-2
#
# Consider the following LO problem:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & 2x_1 + x_2 + 4x_3 & \text{(objective)} \\
# \text{subject to} \quad & x_1 - 2x_2 + 2x_3 \le 120 \\
# & -x_1 - x_2 + 3x_3 = 100 \\
# & x_1 - x_2 + x_3 \ge 80 \\
# & x_i \ge 0 \text{ for all } i.
# \end{aligned}
# $$
#
# a. Solve it in pulp (mind the sign of `>=` and `==` constraints — pulp accepts them directly, unlike some solvers that require everything rewritten as `<=`).
#
# b. Rewrite it in the general form with maximization and "≤" constraints.
#
# c. Solve this problem in pulp and check it gives the same solution as a.
# :::
#
# Sometimes we write the general formulation in matrix notation. Then it becomes:
#
# $$
# \max\{p^T x \mid Ax \le b,\ x \ge 0\},
# $$
#
# where $p$ and $x$ are $n$-dimensional column vectors (and thus $p$ transposed, $p^T$, is a row vector), $b$ is an $m$-dimensional column vector, and $A$ is an $m \times n$ matrix.
#
# Not all LO problems can be solved. Sometimes the problem is unbounded, meaning that solutions of arbitrarily large values can be found. On the other hand, there are also problems where there are no feasible solutions at all. [](#fig-lo-degenerate) shows examples of both situations; pulp reports these as `LpStatus` values `"Unbounded"` and `"Infeasible"` respectively, instead of `"Optimal"`.

# %% [markdown]
# :::{figure} images/lecture8_fig6.2.png
# :label: fig-lo-degenerate
#
# An unbounded (left) and an infeasible (right) LO problem.
# :::

# %% [markdown]
# :::{exercise}
# :label: ex-6-3
#
# Enter both problems shown above in pulp and see what `LpStatus` you get.
# :::

# %% [markdown]
# (lo-larger-example)=
# ## A Larger Example in pulp
#
# Now we show how to solve a slightly larger problem using pulp:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & 2x_1 + 4x_2 + 8x_3 & \text{(objective)} \\
# \text{subject to} \quad & x_1 + 3x_2 + 2x_3 \le 10 & \text{(constraint 1)} \\
# & x_1 + 3x_3 \le 12 & \text{(constraint 2)} \\
# & x_1, x_2, x_3 \ge 0.
# \end{aligned}
# $$
#
# Just like before, we separate the data from the model. This time the coefficients naturally form a matrix (one row per constraint, one column per variable), which we keep as a plain nested list — no separate "cells" for data versus formulas as in a spreadsheet, since in pulp the data and the constraint *expressions* are always kept apart by construction.

# %%
objective_coefs = [2, 4, 8]
constraint_coefs = [[1, 3, 2], [1, 0, 3]]
rhs = [10, 12]

n = len(objective_coefs)
m = len(rhs)

larger_lo = pulp.LpProblem(name="larger_lo", sense=pulp.LpMaximize)
y = [pulp.LpVariable(name=f"x_{j+1}", lowBound=0) for j in range(n)]

larger_lo += pulp.lpSum(objective_coefs[j] * y[j] for j in range(n))
for i in range(m):
    larger_lo += pulp.lpSum(constraint_coefs[i][j] * y[j] for j in range(n)) <= rhs[i]

larger_lo.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[larger_lo.status])
print("optimal solution:", [yj.value() for yj in y])
print("optimal objective:", larger_lo.objective.value())

# %% [markdown]
# This confirms the optimal objective value of 34.67 without ever needing a spreadsheet cell, a formula, or a solver dialog: the same three ingredients as before (data, decision variables, objective and constraints) are enough, and they scale to problems with many more variables and constraints exactly as they are.
#
# :::{exercise}
# :label: ex-6-4
#
# Re-solve the LO problem above by adding the constraints one at a time and checking after each addition how the optimal objective value changes.
# :::
#
# :::{exercise}
# :label: ex-6-5
#
# The tax office can only check a subset of the tax declarations it received. There are 3 types of employees with different skills, and 3 types of declarations. Per declaration type, the expected revenues from additional taxation are as follows: (200, 1000, 500) Euros. Every tax employee can process every declaration, except for employee type 2 who cannot process declaration type 2 and employee type 3 who cannot process declaration type 3. The time per declaration depends on the declaration type and is (1, 3, 2) hours, respectively, except for employee type 3 who takes 2 hours for a type 1 declaration. The numbers of declarations are (15000, 6000, 8000), the numbers of available hours are (10000, 20000, 15000). How do you assign the employees to the different declaration types? Use pulp to solve this problem.
# :::
#
# The CBC solver bundled with pulp is capable enough for problems like these. If you ever run into a bigger instance where CBC is too slow, pulp can call other solvers without changing the model itself — more on choosing a solver in [Modeling Tools and Solvers](lecture10_modeling-tools.ipynb).
#
# :::{exercise}
# :label: ex-6-6
#
# Extend the problem above in the following ways. When solving these problems it helps to think what the additional decision is that needs to be taken.
#
# a. Assume that, next to the 10 units available, you can buy extra units of resource 2 for the price of 1 per unit. What is the optimal solution now?
#
# b. The same question, but now you can buy resource 1 for 2 per unit.
#
# c. The same question, but now you can buy resource 1 for 1 per unit. Can you interpret the result?
# :::

# %% [markdown]
# ## Example Linear Optimization Problems
#
# In this section we consider several types of optimization problems that can be solved using LO.
#
# (graphs-intro)=
# A graph is the mathematical name for a network consisting of nodes and edges connecting the nodes. Nodes are sometimes also called vertices. Edges can be directed or undirected, i.e., uni-directional or bi-directional. Directed edges are often called arcs. Many practical problems can be formulated as problems on graphs. One such problem is project planning.
#
# (project-planning)=
# A project consists of a number of activities, each having a duration. These are the nodes in the graph. Additionally, certain activities require others to finish before they can get started. These precedence relations are modeled as directed edges in the graph. In [](#fig-project-planning) an example of such a graph is given, with each node's duration shown above it. We need to determine the earliest finish time of each activity. These are the decision variables, $x_i$ for activity $i$. Every precedence relation leads to a constraint. When $i$ precedes $j$ then this can be enforced by the constraint $x_i + d_j \le x_j$, where $d_j$ is the duration of activity $j$. For example, in the project below we model the relation $F \to G$ by the constraint $x_F + d_G \le x_G$, i.e., $x_F + 2 \le x_G$.

# %% [markdown]
# :::{figure} images/lecture8_fig6.5.png
# :label: fig-project-planning
#
# Directed graph with weights of a project planning problem.
# :::

# %% [markdown]
# We are interested in the time at which all activities are finished. This can be modeled by an additional variable $z$, bigger than all finish times, that has to be minimized. This leads to the following LO formulation:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & z \\
# \text{subject to} \quad & z \ge x_i \text{ for all vertices } i \\
# & x_i + d_j \le x_j \text{ if } i \text{ precedes } j \\
# & x_i \ge d_i \text{ for all vertices } i.
# \end{aligned}
# $$
#
# The last constraint ensures that no activity starts before time 0. Note that the finish time can also be found using an algorithm without LO, and that this algorithm can be extended to random activity durations. This is relevant in practice because often durations are hard to predict accurately, which is one of the main reasons why, for example, IT projects often finish after the scheduled deadline. Let us solve the LO formulation in pulp for the project of [](#fig-project-planning):

# %%
duration = {"A": 2, "B": 3, "C": 2, "D": 1, "E": 2, "F": 3, "G": 2}
precedences = [
    ("A", "B"),
    ("A", "C"),
    ("B", "E"),
    ("C", "D"),
    ("C", "G"),
    ("D", "E"),
    ("F", "G"),
    ("G", "E"),
]

project = pulp.LpProblem(name="project_planning", sense=pulp.LpMinimize)
finish = {a: pulp.LpVariable(name=f"x_{a}", lowBound=duration[a]) for a in duration}
makespan = pulp.LpVariable(name="z", lowBound=0)

for a in duration:
    project += makespan >= finish[a]
for a, b in precedences:
    project += finish[a] + duration[b] <= finish[b]
project += makespan

project.solve(pulp.PULP_CBC_CMD(msg=False))
print("earliest finish times:", {a: finish[a].value() for a in duration})
print("makespan:", makespan.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-7
#
# Solve the project planning problem above by hand (without LO), by repeatedly picking the activity with no unfinished predecessors and the earliest possible start time. Check that you get the same makespan as pulp.
# :::
#
# A more complicated problem that can be solved using LO is the so-called transportation problem. In this problem we have to transport a single type of good from $n$ sources to $m$ destinations. Source $i$ has supply $a_i$, destination $j$ has demand $b_j$, and link $i \to j$ has transportation costs $c_{ij}$ per unit transported on it. The question is: For each link, how much should you ship on it in order to satisfy the demand of each destination without violating supply constraints? See the figure below for an illustration. When link $i \to j$ does not exist we can take $c_{ij} = \infty$.
#
# The LO formulation is as follows:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij} x_{ij} \\
# \text{subject to} \quad & \sum_{j=1}^{m} x_{ij} \le a_i \text{ for } i = 1, \dots, n; \\
# & \sum_{i=1}^{n} x_{ij} \ge b_j \text{ for } j = 1, \dots, m; \\
# & x_{ij} \ge 0 \text{ for all } i, j.
# \end{aligned}
# $$
#
# :::{exercise}
# :label: ex-6-8
#
# Solve the following transportation problem in pulp, where "x" means there is no connection (use a very large cost, e.g. `1e6`, in place of $\infty$).
# :::

# %% [markdown]
# :::{figure} images/lecture8_fig6.6.png
# :label: fig-transportation
#
# Transportation problem — supply $a_i$ per source $i$, demand $b_j$ per destination $j$, and costs $c_{ij}$ per source-destination pair (source 1: $a_1=10$, costs to destinations 1-4 are 0, 5, x, 0; source 2: $a_2=6$, costs 4, 6, 4, 3; source 3: $a_3=10$, costs 2, 4, 4, 6; demands $b_j$: 5, 5, 5, 5).
# :::

# %% [markdown]
# (transshipment-problem)=
# If we add intermediate nodes to the transportation problem, as in [](#fig-transshipment), then we obtain the transshipment problem. We can solve it by adding constraints of the form:
#
# $$
# \sum_{i=1}^{n} x_{ik} = \sum_{j=1}^{m} x_{kj}
# $$
#
# for all intermediate nodes $k$. It can be extended to a network by adding multiple layers of intermediate nodes.
#
# (production-inventory-model)=
# Next we consider multi-period production/inventory models. Here we assume there are multiple time periods, say $t = 1, \dots, T$, and a starting inventory $s_0$. Every day, the amount of production or supply has to be decided: $x_t$. There is demand, $d_t$ at time $t$, holding costs $h_t$, and production costs $c_t$. The problem is to find the production/order schedule that minimizes the total costs. This can be formulated as an LO problem as follows:

# %% [markdown]
# :::{figure} images/lecture8_fig6.7.png
# :label: fig-transshipment
#
# Transshipment problem.
# :::

# %% [markdown]
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{t=1}^{T} (c_t x_t + h_t s_t) \\
# \text{subject to} \quad & s_{t+1} = s_t - d_{t+1} + x_{t+1} \text{ for } t = 1, \dots, T-1; \\
# & x_t, s_t \ge 0 \text{ for } t = 1, \dots, T.
# \end{aligned}
# $$
#
# This model can be extended in many different directions, such as maximum stock or production capacity, multiple products and resources, backorders, and fixed order costs.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.1 "Problem Formulation", §6.2 "LO in Excel", §6.3 "Example LO Problems." (Excel content replaced with pulp throughout this notebook.)
# - PuLP documentation: https://coin-or.github.io/pulp/
