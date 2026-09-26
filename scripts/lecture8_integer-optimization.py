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
# # Lecture 8: Integer Optimization
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture8_integer-optimization.ipynb)

# %% [markdown]
# An integer linear optimization (ILO) problem is an LO problem with the extra requirement
# that some or all decision variables take integer values, $x_i \in \{0, 1, 2, \dots\}$, or
# are binary, $x_i \in \{0, 1\}$. (A binary variable is just an integer one with the added
# constraint $x_i \le 1$.)
#
# In pulp this is a one-word change: set a variable's `cat` to `"Integer"` or `"Binary"`
# instead of the default `"Continuous"`. Everything else about building and solving the
# model is the same. Solving it, however, is a different matter: in general ILO is much
# harder than LO, as this notebook and [Complexity](lecture11_complexity.ipynb) explain.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - recognize when a problem needs integer or binary decision variables, and formulate and
#   solve it in pulp;
# - explain why ILO is generally harder to solve than LO;
# - explain how branch and bound finds the optimum of an ILO problem;
# - write any LO problem in the general matrix form;
# - explain why linearity (but not integrality) is essential for efficient solvability, and
#   how integer variables let you model many other nonlinearities without giving that up.

# %% [markdown]
# ## Why Integer Problems Are Harder
#
# Take the bookcase/desk product-mix problem from
# [Linear Optimization](lecture8_linear-optimization.ipynb) and require whole units of $x$
# and $y$. Its LO optimum was $(x, y) = (3.6, 2.8)$ with profit 24.8, not integer. Two
# things go wrong compared to LO:
#
# - **the optimal corner is no longer feasible**, so the simplex reasoning ("the optimum is
#   at a corner") does not directly help;
# - **rounding the LO optimum is not enough**: rounding both up to $(4, 3)$ violates the
#   oak-panel constraint ($4 + 9 = 13 > 12$), and rounding both down to $(3, 2)$ is feasible
#   but only reaches a profit of 19, well below the true integer optimum. Evaluating nearby
#   integer points tells us little, because finding the true optimum in general needs a
#   systematic search.
#
# Let pulp solve the integer version:

# %%
import matplotlib.pyplot as plt
import numpy as np
import pulp

profit = {"bookcase": 3, "desk": 5}
# resource_use[res][p]: units of resource res per unit of product p
resource_use = {
    "oak panels": {"bookcase": 1, "desk": 3},
    "assembly hours": {"bookcase": 2, "desk": 1},
}
available = {"oak panels": 12, "assembly hours": 10}
products = list(profit)

int_mix = pulp.LpProblem(name="integer_product_mix", sense=pulp.LpMaximize)
dec_vars = {
    p: pulp.LpVariable(name=p, lowBound=0, cat="Integer") for p in products
}
int_mix += pulp.lpSum(profit[p] * dec_vars[p] for p in products)
for res, cap in available.items():
    usage = pulp.lpSum(resource_use[res][p] * dec_vars[p] for p in products)
    int_mix += usage <= cap, res.replace(" ", "_")
solver = pulp.getSolver("COIN_CMD", msg=False)
int_mix.solve(solver)
print("integer optimum:", {p: dec_vars[p].value() for p in products})
print("optimal profit:", int_mix.objective.value())

# %% [markdown]
# The picture below shows why rounding is unreliable: the LP-relaxation optimum (the star)
# does not sit on the integer grid, and the nearest lattice points are not necessarily
# feasible or optimal. The ILO optimum (the black dot) is the best *feasible* grid point,
# which can be several steps away from the naive rounding of the relaxation.

# %% tags=["hide-input"]
x_grid = np.linspace(0, 6, 200)
plt.figure(figsize=(5, 5))
plt.plot(
    x_grid,
    (12 - x_grid) / 3,
    color="C0",
    label=r"$x + 3y \leq 12$ (oak panels)",
)
plt.plot(
    x_grid,
    10 - 2 * x_grid,
    color="C1",
    label=r"$2x + y \leq 10$ (assembly hours)",
)
y_upper = np.minimum((12 - x_grid) / 3, 10 - 2 * x_grid)
plt.fill_between(
    x_grid,
    0,
    y_upper,
    where=(y_upper >= 0),
    alpha=0.15,
    label="feasible region",
)

xs, ys = np.meshgrid(range(7), range(5))
feasible = (xs + 3 * ys <= 12) & (2 * xs + ys <= 10)
plt.scatter(
    xs[feasible],
    ys[feasible],
    color="C2",
    zorder=3,
    label="integer feasible points",
)
plt.scatter(
    xs[~feasible],
    ys[~feasible],
    color="lightgrey",
    zorder=2,
    label="integer infeasible points",
)

plt.plot(
    3.6, 2.8, "C3*", markersize=14, zorder=4, label="LP relaxation optimum"
)
x_int, y_int = dec_vars["bookcase"].value(), dec_vars["desk"].value()
plt.plot(x_int, y_int, "ko", markersize=8, zorder=4, label="ILO optimum")

plt.xlim(0, 6)
plt.ylim(0, 4.5)
plt.xlabel("bookcases $x$")
plt.ylabel("desks $y$")
plt.legend(loc="upper right", fontsize=7)
plt.title("Feasible region with integer lattice points")
plt.show()

# %% [markdown]
# :::{exercise}
# :label: ex-6-9
#
# Solve the larger LO problem from
# [Linear Optimization](lecture8_linear-optimization.ipynb#larger-lo-example) again, once
# requiring all variables integer and once requiring them binary. Compare the optimal
# objective values with the continuous one.
# :::

# %% [markdown]
# ## Branch and Bound
#
# A common method for solving integer linear optimization (ILO) problems is **branch and bound**. The basic idea is to start by just ignoring the integer restrictions. More precisely, we allow integer variables to be continuous while retaining all their other bounds and constraints. For example, instead of requiring a binary variable to satisfy $x_i \in \{0,1\}$, we allow it to take any value between zero and one:
# $$
# 0 \leq x_i \leq 1.
# $$
# Likewise, instead of requiring $x_i \in \{0,1,2,\ldots\}$, we only require:
# $$
# x_i \geq 0.
# $$
# This produces the **LO relaxation**: the same optimization problem, but without the requirement that certain variables must be integers.
#
# The optimal solution to the LO relaxation usually contains non-integer values that should be integers. Such a solution is therefore infeasible for the original ILO problem, but it is still useful in two ways. First, for a maximization problem, it provides an **upper bound (UB)**: the best possible integer solution cannot have a higher objective value than the relaxed solution. Second, it provides a starting point for further search by **branching** on non-integer variables one by one.
#
# Branching on an LO relaxation that yields an infeasible ILO solution works as follows. Suppose the relaxed solution contains $x_j = 2.5$, while $x_j$ should be an integer. We then branch the original root problem into two subproblems based on the LO relaxation of the original problem:
#
# - First subproblem: original problem but with extra constraint $x_j \leq 2$;
# - Second subproblem: original problem but with extra constraint $x_j \geq 3$;
#
# Together, these two subproblems cover all possible integer values of $x_j$, but neither allows $x_j = 2.5$ anymore. We then solve the LO relaxation of each subproblem (independently) in the same manner. If either relaxation again yields a non-integer variable that must be integer, we apply the same strategy and branch again, et cetera. To illustrate this, suppose that solving the relaxation for the subproblem with $x_j \leq 2$ leads to a solution with $x_k = 5.6$. We then branch this subproblem again to get one subproblem with constraints $x_j \leq 2$ and $x_k \leq 5$, and one subproblem with $x_j \leq 2$ and $x_k \geq 6$. Indeed, both subproblems still have the previous constraint $x_j \leq 2$.
#
# At first sight, this may seem like enumerating all possible integer values. The crucial difference is that branch and bound uses bounds to avoid exploring subproblems that cannot improve the best integer solution found so far. Hence the name branch and bound.
#
# Any feasible integer solution to the original ILO problem provides a **lower bound (LB)** for a maximization problem. Along the way, we keep track of the best original ILO solution, called the **incumbent**. Its objective value is the current best/largest LB. On the other hand, as mentioned before, the LO relaxation of a subproblem is always an UB for the best integer-feasible solution at the subproblem. So if the UB at a subproblem is smaller than the current best LB, we don't need to branch further on that subproblem, as it will never lead to a better ILO solution, saving computation time.
#
# Let us formalize this method further. The branch and bound method keeps track of (1) a pool of (sub)problems whose LO relaxation is not solved yet and which may still lead to an optimal solution, and (2) the current best solution with the largest/best LB so far. Initially, this pool contains only the original problem, and the LB is set to $-\infty$. As long as the pool is not empty, choose a (sub)problem from the pool, remove it, and solve its LO relaxation. This solve leads to two possibilities:
#
# 1. The found solution is infeasible for the ILO problem, and its objective value is an UB:
#    1. If UB $\leq$ largest LB: We eliminate this subproblem (we will not find a better solution here; this can by default not happen for the original problem).
#    2. If UB $>$ largest LB: Pick a variable from the solution that is non-integer but should be. Branch the (sub)problem on this variable into two new (sub)problems and add them to the pool.
# 2. The found solution is feasible for the ILO problem and gives a new LB:
#    1. If LB $>$ largest LB: Update the new best LB and see whether we can eliminate subproblems that were not eliminated yet. In particular, for all non-eliminated subproblems for which we obtained a UB, check whether their UB $<$ largest LB; if so, eliminate the subproblem and remove all its descendants still in the pool.
#    2. If LB $\leq$ largest LB: Eliminate this subproblem (nothing to branch further here).
# 3. There is no feasible solution for the specific (sub)problem: Eliminate this subproblem as it cannot have an ILO feasible solution.
#
# Keep repeating this procedure until the pool is empty. Once the pool is empty, we are guaranteed that the solution with the current best LB is optimal.
#
# The process can be visualized as a tree. The original problem is the root. Each time we branch, we split one (sub)problem into two child subproblems. The pool contains the subproblems that have not yet been eliminated or solved.
#
# We did not discuss how to choose two things: which decision variable to branch on (since an LO relaxation solution typically has many non-integer variables that should be integer) and which (sub)problem from the pool to pick first. The choice affects performance, and what works best is a research topic on its own and outside the scope of this course. In this course, just make a choice; we always know that in the end we will find the optimal solution.
#
# Let us apply this idea to the integer product-mix problem to illustrate its workings:
#
# - **Root problem.** The LO relaxation has optimum $\left(3.6, 2.8\right)$ with objective value $24.8$. This is an UB, but the solution is not integer in both decision variables and thus infeasible. Just pick a decision variable to branch on. We branch on $y = 2.8$.
# - **Left subproblem:** root problem with $y \leq 2$. Its relaxation has optimum $\left(4, 2\right)$ with value $22$. This solution is integer, so it is feasible for the ILO problem and thus holds the current best LB of $22$. No need to branch further from this subproblem.
# - **Right subproblem:** root problem with $y \geq 3$. Its relaxation has optimum $\left(3, 3\right)$ with value $24$. This is also feasible for the ILO problem, so it improves the best LB to $24$, and we can replace the previous best solution with the solution $\left(3, 3\right)$. No need to branch further from this subproblem.
#
# Since the pool is now empty (no promising (sub)problems left to explore), we are done, and the current best solution of $\left(3,3\right)$ with profit $24$ must be the optimal ILO solution.
#
# In this example, we needed only three LO relaxations, rather than checking every possible integer combination. Modern integer-optimization solvers use this basic branch-and-bound idea, often enhanced with additional techniques. More on this later.

# %% [markdown]
# ## The Knapsack Problem
#
# The archetypal binary ILO problem is the knapsack problem: from a set of items, each with
# a *reward* and a *weight*, choose a subset of maximum total reward whose total weight fits
# a given capacity. Applications include which items to load in a truck, cutting stock in a steel
# plant, and simple forms of portfolio selection.
#
# Let us consider a concrete example with $n=6$ numbered items:
#
# | item $i$     | 1 | 2 | 3 | 4 | 5 | 6 |
# |--------------|---|---|---|---|---|---|
# | reward $r_i$ | 10 | 13 | 18 | 31 | 7 | 15 |
# | weight $w_i$ | 2 | 3 | 4 | 7 | 1 | 3 |
#
# In here, the reward and weight of the $i$th item is denoted by $r_i$ and $w_i$, respectively. The capacity $C = 10$. Furthermore, define the binary decision variable $x_i$ as 1 if we take item $i$ and 0 otherwise. The ILO problem can then be stated as follows:
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_{i=1}^n r_i x_i \\
# \text{subject to} \quad & \sum_{i=1}^n w_i x_i \le C \\
# & x_i \in \{0, 1\} \text{ for all } i.
# \end{aligned}
# $$
#
# This problem can be solved in pulp as follows.

# %%
# data
reward = [10, 13, 18, 31, 7, 15]
weight = [2, 3, 4, 7, 1, 3]
capacity = 10
n_items = len(reward)

# modeling
knapsack = pulp.LpProblem(name="knapsack", sense=pulp.LpMaximize)
take = [
    pulp.LpVariable(name=f"x_{i + 1}", cat="Binary") for i in range(n_items)
]
knapsack += pulp.lpSum(reward[i] * take[i] for i in range(n_items))
knapsack += pulp.lpSum(weight[i] * take[i] for i in range(n_items)) <= capacity

# solve and print solution
solver = pulp.getSolver("COIN_CMD", msg=False)
knapsack.solve(solver)
print("take items:", [i + 1 for i in range(n_items) if take[i].value() == 1])
print("total reward:", knapsack.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-10
#
# Take the knapsack solution pulp found above. Verify by hand that no single swap (adding
# one currently-excluded item and removing whatever is needed to stay within capacity)
# improves the total reward.
# :::
#
# :::{exercise}
# :label: ex-6-11
#
# Solve, by branch and bound *on paper*, the knapsack problem with rewards $(15, 9, 10, 5)$,
# weights $(1, 3, 5, 4)$ and capacity 8. For the relaxation bound, fill the capacity with
# items in decreasing order of reward-to-weight ratio, allowing a fraction of the last one.
# Check your answer with pulp.
# :::

# %% [markdown]
# (general-formulation)=
# ## General Formulation
#
# Having now seen both LO and ILO in action, we can step back and write down the general
# form both fit into. An LO problem with $n$ decision variables and $m$ constraints is
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_{j=1}^{n} p_j x_j \\
# \text{subject to} \quad & \sum_{j=1}^{n} a_{ij} x_j \le b_i, \quad i = 1, \dots, m \\
# & x_1, \dots, x_n \ge 0,
# \end{aligned}
# $$
#
# or in matrix notation $\max\{p^T x \mid A x \le b,\ x \ge 0\}$, with $p, x$ column vectors
# of length $n$, $b$ a column vector of length $m$, and $A$ an $m \times n$ matrix.
#
# This one form covers more than it seems. Every other case can be rewritten into it:
#
# - **minimization**: $\min p^T x = -\max (-p^T x)$;
# - **"$\ge$" constraints**: $Ax \ge b \Leftrightarrow -Ax \le -b$;
# - **"$=$" constraints**: $Ax = b \Leftrightarrow Ax \le b$ and $Ax \ge b$;
# - **free (unrestricted) variables**: replace $x$ by $x^+ - x^-$ with $x^+, x^- \ge 0$.

# %% [markdown]
# (why-linearity-matters)=
# ## Why Linearity Matters, and What Integrality Buys Back
#
# Linearity is what makes LO efficiently solvable: written in the
# [general form above](#general-formulation), the feasible region is a convex polyhedron,
# an optimum sits at a corner, and a local optimum is automatically global. As soon as the
# objective or a constraint is nonlinear, both of those break:
#
# - **Nonlinear objective.** Maximize $x_1 x_2$ subject to $x_1 + x_2 \le 1$,
#   $x_1, x_2 \ge 0$. The optimum is $(0.5, 0.5)$, in the *interior* of an edge, not at a
#   corner.
# - **Nonlinear constraint.** Maximize $x_1 + x_2$ subject to $\min(x_1, x_2) = 0$,
#   $x_1 \le 2$, $x_2 \le 1$. The feasible set is two line segments meeting at the origin
#   (either $x_1 = 0$ or $x_2 = 0$). It has two *local* optima, $(2, 0)$ and $(0, 1)$; you
#   cannot be sure which is global without checking both. More generally, the simplex
#   reasoning from [Linear Optimization](lecture8_linear-optimization.ipynb) breaks down
#   whenever the feasible region is not a convex polyhedron: you are not sure you have
#   found the best solution until you have checked every local optimum, which is usually
#   intractable.
#
# Nonlinear optimization therefore needs slower, less reliable algorithms. But there is a
# large and useful middle ground: integer constraints. On the one hand, requiring
# $x_i \in \{0, 1, 2, \dots\}$ is itself a nonlinear constraint, and it does make a problem
# harder to solve, as branch and bound's extra work above shows. On the other hand, unlike
# general nonlinearities, it is not *intractably* so, and many other nonlinearities (an
# either/or choice, a fixed cost that applies only when an activity is used, a "this
# constraint holds only if..." condition) can be expressed with integer (usually binary)
# variables and otherwise-linear constraints, and then solved with branch and bound. That is
# why so much modeling effort goes into casting a problem as ILO, and why it is worth
# treating as its own class rather than lumping it in with general nonlinear optimization.
# [Set Covering and Shift Scheduling](lecture9_covering.ipynb) and
# [Machine Scheduling](lecture9_machine-scheduling.ipynb) show many of those tricks in
# practice.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.4 "Integer Problems."
